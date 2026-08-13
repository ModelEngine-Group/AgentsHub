"""paramiko 封装: 支持直连(UCloud 密码)与 OpenLibing ProxyJump+密钥。

用法:
  python scripts/_ssh_npu.py run  "<remote command>"
  python scripts/_ssh_npu.py put  <local> <remote>
  python scripts/_ssh_npu.py get  <remote> <local>

凭据从 .env 读取(NPU_SSH_*),本文件不含明文凭据。

直连模式（旧）:
  NPU_SSH_HOST / PORT / USER / PASSWORD / HOST_KEY_SHA256

跳板模式（密钥信息/OpenLibing）:
  NPU_SSH_JUMP_HOST / JUMP_PORT / JUMP_USER
  NPU_SSH_HOST / PORT / USER
  NPU_SSH_KEY_FILE
  NPU_SSH_HOST_KEY_SHA256   # 目标机 host key
  可选 NPU_SSH_JUMP_HOST_KEY_SHA256
"""
from __future__ import annotations

import base64
import hmac
import os
import pathlib
import sys
import time as _time

import paramiko
from dotenv import load_dotenv

ROOT = pathlib.Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

HOST = os.environ["NPU_SSH_HOST"]
PORT = int(os.environ.get("NPU_SSH_PORT", "22"))
USER = os.environ["NPU_SSH_USER"]
PASSWORD = os.environ.get("NPU_SSH_PASSWORD", "")
HOST_KEY_SHA256 = os.environ.get("NPU_SSH_HOST_KEY_SHA256", "").strip()
KEY_FILE = os.environ.get("NPU_SSH_KEY_FILE", "").strip()
JUMP_HOST = os.environ.get("NPU_SSH_JUMP_HOST", "").strip()
JUMP_PORT = int(os.environ.get("NPU_SSH_JUMP_PORT", "22"))
JUMP_USER = os.environ.get("NPU_SSH_JUMP_USER", "jump")
JUMP_HOST_KEY_SHA256 = os.environ.get("NPU_SSH_JUMP_HOST_KEY_SHA256", "").strip()

_OPEN_RETRIES = int(os.environ.get("NPU_SSH_OPEN_RETRIES", "6"))
_OPEN_BACKOFF = float(os.environ.get("NPU_SSH_OPEN_BACKOFF", "4"))


def _normalize_fingerprint(value: str) -> str:
    return value if value.startswith("SHA256:") else f"SHA256:{value}"


class _PinnedHostKeyPolicy(paramiko.MissingHostKeyPolicy):
    def __init__(self, expected: str):
        self._expected = _normalize_fingerprint(expected)

    def missing_host_key(self, client, hostname, key):
        actual = key.fingerprint
        if not hmac.compare_digest(actual, self._expected):
            raise RuntimeError(
                f"SSH host-key mismatch for {hostname}: expected {self._expected}, received {actual}"
            )


def _pkey():
    if not KEY_FILE:
        return None
    path = pathlib.Path(KEY_FILE)
    if not path.is_file():
        raise RuntimeError(f"NPU_SSH_KEY_FILE not found: {KEY_FILE}")
    return paramiko.RSAKey.from_private_key_file(str(path))


def _attach_policy(client: paramiko.SSHClient, expected: str) -> None:
    if expected:
        client.set_missing_host_key_policy(_PinnedHostKeyPolicy(expected))
    else:
        raise RuntimeError("host-key pin required; refusing unpinned SSH host")


def connect() -> paramiko.SSHClient:
    if not HOST_KEY_SHA256:
        raise RuntimeError(
            "NPU_SSH_HOST_KEY_SHA256 is required; refusing an unpinned SSH host"
        )

    if JUMP_HOST:
        # ProxyJump: jump --direct-tcpip--> target
        if not KEY_FILE:
            raise RuntimeError("ProxyJump mode requires NPU_SSH_KEY_FILE")
        pkey = _pkey()
        jump = paramiko.SSHClient()
        _attach_policy(jump, JUMP_HOST_KEY_SHA256 or HOST_KEY_SHA256)
        # If jump pin differs, prefer JUMP pin; if only target pin set, AutoAdd jump
        # is unsafe — require explicit jump pin when using jump.
        if not JUMP_HOST_KEY_SHA256:
            raise RuntimeError(
                "NPU_SSH_JUMP_HOST_KEY_SHA256 is required in ProxyJump mode"
            )
        jump.set_missing_host_key_policy(_PinnedHostKeyPolicy(JUMP_HOST_KEY_SHA256))
        jump.connect(
            JUMP_HOST,
            port=JUMP_PORT,
            username=JUMP_USER,
            pkey=pkey,
            look_for_keys=False,
            allow_agent=False,
            timeout=30,
            banner_timeout=30,
            auth_timeout=30,
        )
        channel = jump.get_transport().open_channel(
            "direct-tcpip",
            (HOST, PORT),
            ("127.0.0.1", 0),
            timeout=30,
        )
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(_PinnedHostKeyPolicy(HOST_KEY_SHA256))
        client.connect(
            hostname=HOST,
            port=PORT,
            username=USER,
            pkey=pkey,
            sock=channel,
            look_for_keys=False,
            allow_agent=False,
            timeout=30,
            banner_timeout=30,
            auth_timeout=30,
        )
        # Keep jump client alive by stashing on the target client.
        client._dki_jump_client = jump  # type: ignore[attr-defined]
        return client

    # Direct password mode (legacy UCloud gateway)
    if not PASSWORD:
        raise RuntimeError("direct mode requires NPU_SSH_PASSWORD")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(_PinnedHostKeyPolicy(HOST_KEY_SHA256))
    client.connect(
        HOST,
        port=PORT,
        username=USER,
        password=PASSWORD,
        timeout=30,
        banner_timeout=30,
        auth_timeout=30,
        look_for_keys=False,
        allow_agent=False,
    )
    return client


def _close_client(client: paramiko.SSHClient) -> None:
    jump = getattr(client, "_dki_jump_client", None)
    try:
        client.close()
    except BaseException:
        pass
    if jump is not None:
        try:
            jump.close()
        except BaseException:
            pass


def _open_session_with_retry(make_chan):
    last = None
    for attempt in range(_OPEN_RETRIES):
        client = connect()
        try:
            transport = client.get_transport()
            chan = transport.open_session()
            return client, chan
        except (EOFError, paramiko.SSHException, OSError) as exc:
            last = exc
            _close_client(client)
            _time.sleep(_OPEN_BACKOFF * (attempt + 1))
    raise RuntimeError(f"open_session failed after {_OPEN_RETRIES} retries: {last}")


def run(cmd: str, timeout: int = 1800) -> int:
    client, chan = _open_session_with_retry(None)
    try:
        chan.settimeout(timeout)
        chan.exec_command(f"bash -lc {shquote(cmd)}")
        while True:
            if chan.recv_ready():
                sys.stdout.write(chan.recv(65536).decode("utf-8", "replace"))
                sys.stdout.flush()
            if chan.recv_stderr_ready():
                sys.stderr.write(chan.recv_stderr(65536).decode("utf-8", "replace"))
                sys.stderr.flush()
            if chan.exit_status_ready() and not chan.recv_ready() and not chan.recv_stderr_ready():
                break
        return chan.recv_exit_status()
    finally:
        _close_client(client)


def shquote(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"


def put(local: str, remote: str) -> None:
    """经 exec 通道以 base64 流上传（兼容无 SFTP 网关）。"""
    data = pathlib.Path(local).read_bytes()
    client, chan = _open_session_with_retry(None)
    try:
        chan.exec_command(f"base64 -d > {shquote(remote)}")
        chan.sendall(base64.b64encode(data))
        chan.shutdown_write()
        rc = chan.recv_exit_status()
        if rc != 0:
            raise RuntimeError(f"remote write failed rc={rc}: {chan.recv_stderr(4096).decode()}")
        print(f"put {local} -> {remote} ({len(data)} bytes)")
    finally:
        _close_client(client)


def get(remote: str, local: str) -> None:
    client, chan = _open_session_with_retry(None)
    try:
        chan.exec_command(f"base64 {shquote(remote)}")
        out, err = chan.makefile("r"), chan.makefile_stderr("r")
        data = base64.b64decode(out.read())
        e = err.read().decode()
        if e.strip():
            raise RuntimeError(f"remote read failed: {e}")
        pathlib.Path(local).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(local).write_bytes(data)
        print(f"get {remote} -> {local} ({len(data)} bytes)")
    finally:
        _close_client(client)


def main() -> int:
    mode = sys.argv[1]
    if mode == "run":
        return run(sys.argv[2])
    if mode == "put":
        put(sys.argv[2], sys.argv[3])
    elif mode == "get":
        get(sys.argv[2], sys.argv[3])
    else:
        raise SystemExit(f"unknown mode: {mode}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
