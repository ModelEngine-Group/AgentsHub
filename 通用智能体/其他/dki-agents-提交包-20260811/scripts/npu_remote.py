"""NPU 服务器远程操作工具（paramiko 封装）。

背景：该服务器的 SSH 网关只支持密码认证（publickey 已实测被拒），
plain ssh/scp 无法非交互使用，故所有远端操作统一走本工具。

凭据：从项目根 .env 读取（该文件在 .gitignore 中，不会随仓库交付）：
    NPU_SSH_HOST=...
    NPU_SSH_PORT=...
    NPU_SSH_USER=...
    NPU_SSH_PASSWORD=...

用法（本地项目根目录）：
    python scripts/npu_remote.py run "npu-smi info"          # bash -lc 下执行，实时回显
    python scripts/npu_remote.py put 本地文件 /root/目标路径
    python scripts/npu_remote.py get /root/远端路径 本地目录   # 远端目录自动递归
"""

import hmac
import socket
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_env():
    env = {}
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    required = (
        "NPU_SSH_HOST",
        "NPU_SSH_PORT",
        "NPU_SSH_USER",
        "NPU_SSH_PASSWORD",
        "NPU_SSH_HOST_KEY_SHA256",
    )
    missing = [key for key in required if not env.get(key)]
    if missing:
        sys.exit(f"[!] .env 缺少配置项: {missing}（见 HANDOFF_NPU_GPT5.md 第 2 节）")
    return env


def connect():
    import paramiko

    env = load_env()
    host = env["NPU_SSH_HOST"]
    port = int(env["NPU_SSH_PORT"])
    expected = env["NPU_SSH_HOST_KEY_SHA256"]
    if not expected.startswith("SHA256:"):
        expected = f"SHA256:{expected}"

    sock = socket.create_connection((host, port), timeout=30)
    transport = paramiko.Transport(sock)
    client = paramiko.SSHClient()
    try:
        transport.start_client(timeout=30)
        actual = transport.get_remote_server_key().fingerprint
        if not hmac.compare_digest(actual, expected):
            raise RuntimeError(
                f"SSH host-key mismatch: expected {expected}, received {actual}"
            )
        transport.auth_password(env["NPU_SSH_USER"], env["NPU_SSH_PASSWORD"])
        if not transport.is_authenticated():
            raise RuntimeError("SSH password authentication failed")
        client._transport = transport
    except BaseException:
        transport.close()
        sock.close()
        raise
    return client


def cmd_run(client, command: str) -> int:
    """bash -lc 下执行远端命令，实时回显，返回退出码。

    注意：走 SSHClient.exec_command（不申请 PTY）——该网关对手动
    open_session+get_pty 的通道协商会异常掐断（实测），exec_command 路径稳定。
    """
    import shlex

    stdin, stdout, stderr = client.exec_command(
        f"bash -lc {shlex.quote(command)}", timeout=None
    )
    chan = stdout.channel
    while True:
        while chan.recv_ready():
            sys.stdout.write(chan.recv(4096).decode("utf-8", "replace"))
            sys.stdout.flush()
        while chan.recv_stderr_ready():
            sys.stdout.write(chan.recv_stderr(4096).decode("utf-8", "replace"))
            sys.stdout.flush()
        if chan.exit_status_ready() and not chan.recv_ready() and not chan.recv_stderr_ready():
            break
        time.sleep(0.2)
    code = chan.recv_exit_status()
    print(f"\n[exit={code}]")
    return code


def cmd_put(client, local: str, remote: str) -> int:
    """上传单个文件（exec + base64 管道；该网关不支持 SFTP 子系统）。

    目录请先 tar 成单文件再 put。
    """
    import base64
    import shlex

    local_p = Path(local)
    if not local_p.is_file():
        sys.exit(f"[!] 本地文件不存在: {local}（目录请先 tar 成单文件再 put）")
    data = base64.b64encode(local_p.read_bytes())
    stdin, stdout, stderr = client.exec_command(
        f"base64 -d > {shlex.quote(remote)} && echo PUT_OK", timeout=None
    )
    # 分块写入 stdin，写完关闭写端
    for i in range(0, len(data), 32768):
        stdin.write(data[i:i + 32768])
    stdin.channel.shutdown_write()
    out = stdout.read().decode().strip()
    err = stderr.read().decode().strip()
    if "PUT_OK" not in out:
        sys.exit(f"[!] 上传失败: {err or out}")
    print(f"[√] {local} → {remote} ({local_p.stat().st_size} bytes)")
    return 0


def cmd_get(client, remote: str, local: str) -> int:
    """下载远端文件或目录（exec + base64；目录自动 tar 打包递归）。"""
    import base64
    import io
    import shlex
    import tarfile

    local_dir = Path(local)
    local_dir.mkdir(parents=True, exist_ok=True)
    # 判断远端是文件还是目录
    _, stdout, _ = client.exec_command(
        f"test -d {shlex.quote(remote)} && echo DIR || echo FILE", timeout=60)
    kind = stdout.read().decode().strip()

    if kind == "DIR":
        parent = remote.rstrip("/").rsplit("/", 1)[0] or "/"
        name = remote.rstrip("/").split("/")[-1]
        _, stdout, stderr = client.exec_command(
            f"cd {shlex.quote(parent)} && tar czf - {shlex.quote(name)} | base64", timeout=None)
        b64 = stdout.read()
        err = stderr.read().decode().strip()
        raw = base64.b64decode(b64)
        if not raw:
            sys.exit(f"[!] 下载失败: {err or '远端无数据'}")
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tf:
            tf.extractall(local_dir)  # noqa: S202 受控来源
        print(f"[√] 目录 {remote} → {local_dir / name} ({len(raw)} bytes 压缩)")
    else:
        _, stdout, stderr = client.exec_command(
            f"base64 < {shlex.quote(remote)}", timeout=None)
        b64 = stdout.read()
        err = stderr.read().decode().strip()
        raw = base64.b64decode(b64)
        if not raw and err:
            sys.exit(f"[!] 下载失败: {err}")
        dest = local_dir / remote.rstrip("/").split("/")[-1]
        dest.write_bytes(raw)
        print(f"[√] {remote} → {dest} ({len(raw)} bytes)")
    return 0


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    action = sys.argv[1]
    # 网关偶发掐断新建会话（EOFError/SSHException），统一重试 3 次
    last_err = None
    for attempt in range(1, 4):
        client = None
        try:
            client = connect()
            if action == "run":
                sys.exit(cmd_run(client, sys.argv[2]))
            if action == "put":
                sys.exit(cmd_put(client, sys.argv[2], sys.argv[3]))
            if action == "get":
                sys.exit(cmd_get(client, sys.argv[2], sys.argv[3]))
            print(__doc__)
            sys.exit(2)
        except SystemExit:
            raise
        except Exception as e:
            last_err = e
            print(f"[!] 第 {attempt}/3 次尝试失败: {type(e).__name__}: {e}，5 秒后重试...")
            time.sleep(5)
        finally:
            if client is not None:
                try:
                    client.close()
                except Exception:
                    pass
    sys.exit(f"[!] 连续 3 次失败，最后错误: {last_err}")


if __name__ == "__main__":
    main()
