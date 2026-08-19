"""Create-once, isolated demo runs.

The Streamlit front end deliberately delegates the real pipeline to the
existing CLI in a child process.  This keeps the demo on the same production
path as ``demo-all`` while allowing every browser run to use its own
``DKI_OUTPUT_DIR``.
"""

from __future__ import annotations

import calendar
import csv
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


_RUN_LOCK = threading.Lock()
_EVENT_LOCK = threading.Lock()
_SOURCE_TREE_RE = re.compile(r"[0-9a-fA-F]{64}")
_TERMINAL_NAMES = {"completed.json": "completed", "failed.json": "failed"}


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    """Load strict JSON and reject duplicate keys and non-finite numbers."""

    try:
        payload = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON value: {value}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label} is unreadable or invalid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _write_once(path: Path, payload: Any) -> None:
    """Atomically publish one JSON object without exposing a partial file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            # A hard link is an atomic create-if-absent operation on both
            # Windows/NTFS and Linux.  Unlike rename/replace, it cannot
            # overwrite an existing receipt.
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(f"demo receipt already exists: {path}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _terminal_payload(run_dir: Path) -> dict[str, Any] | None:
    """Return the one authoritative terminal receipt, rejecting conflicts."""

    run_dir = Path(run_dir)
    paths = [run_dir / "terminal.json", run_dir / "completed.json", run_dir / "failed.json"]
    present = [path for path in paths if path.is_file()]
    if not present:
        return None
    loaded = [_load_json_object(path, label=f"demo terminal receipt {path.name}") for path in present]
    statuses = {str(payload.get("status")) for payload in loaded}
    if len(statuses) != 1 or statuses.pop() not in {"completed", "failed"}:
        raise ValueError("demo run has conflicting or invalid terminal receipts")
    canonical = json.dumps(loaded[0], ensure_ascii=False, sort_keys=True, allow_nan=False)
    if any(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False) != canonical
        for payload in loaded[1:]
    ):
        raise ValueError("demo run terminal receipts do not match")
    return loaded[0]


def _publish_terminal(run_dir: Path, name: str, payload: dict[str, Any]) -> None:
    """Claim the run's single terminal state, then publish its public alias."""

    if name not in _TERMINAL_NAMES or payload.get("status") != _TERMINAL_NAMES[name]:
        raise ValueError("invalid demo terminal receipt name/status")
    run_dir = Path(run_dir)
    existing = _terminal_payload(run_dir)
    if existing is not None:
        raise FileExistsError(
            f"demo terminal state already published as {existing.get('status')}"
        )
    canonical = run_dir / "terminal.json"
    _write_once(canonical, payload)
    alias = run_dir / name
    try:
        os.link(canonical, alias)
    except FileExistsError as exc:
        # The canonical claim is authoritative.  An independently-created
        # alias is only acceptable when byte-for-byte identical.
        if alias.read_bytes() != canonical.read_bytes():
            raise ValueError("demo terminal alias conflicts with canonical receipt") from exc


@dataclass(frozen=True)
class DemoRunReceipt:
    """Small public receipt consumed by the UI and final evidence exporter."""

    run_id: str
    run_dir: Path
    status: str
    returncode: int | None
    started_at: str
    completed_at: str | None
    source_tree_sha256: str
    mode: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "status": self.status,
            "returncode": self.returncode,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "source_tree_sha256": self.source_tree_sha256,
            "mode": self.mode,
        }


class DemoRunService:
    """Run ``demo-all`` exactly once in a new output directory."""

    def __init__(self, root: Path, *, python: str | None = None, timeout: int = 900):
        self.root = Path(root).resolve()
        self.python = python or sys.executable
        self.timeout = int(timeout)
        if self.timeout <= 0:
            raise ValueError("demo timeout must be a positive number of seconds")
        self.server_rid = self.root.name
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "runs").mkdir(parents=True, exist_ok=True)
        self.reconcile_orphans()

    def new_run_id(self) -> str:
        return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:12]

    def list_runs(self) -> list[Path]:
        return sorted(
            (path for path in (self.root / "runs").iterdir() if path.is_dir()),
            key=lambda path: path.name,
            reverse=True,
        )

    @staticmethod
    def _pid_alive(pid: object) -> bool:
        if type(pid) is not int or pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except (OSError, ValueError):
            return False
        return True

    @staticmethod
    def _pid_start_token(pid: object) -> str | None:
        """Return an OS process-creation token used to detect PID reuse."""

        if type(pid) is not int or pid <= 0:
            return None
        proc_stat = Path(f"/proc/{pid}/stat")
        try:
            text = proc_stat.read_text(encoding="ascii")
            close = text.rfind(")")
            fields = text[close + 2 :].split()
            # The suffix starts at field 3 (state); index 19 is field 22
            # (process start time in clock ticks since boot).
            if close > 0 and len(fields) > 19:
                return f"proc:{fields[19]}"
        except (OSError, UnicodeError, ValueError):
            pass
        if os.name == "nt":
            try:
                import ctypes
                from ctypes import wintypes

                process = ctypes.windll.kernel32.OpenProcess(0x0400, False, pid)
                if not process:
                    return None
                creation = wintypes.FILETIME()
                exit_time = wintypes.FILETIME()
                kernel = wintypes.FILETIME()
                user = wintypes.FILETIME()
                try:
                    ok = ctypes.windll.kernel32.GetProcessTimes(
                        process,
                        ctypes.byref(creation),
                        ctypes.byref(exit_time),
                        ctypes.byref(kernel),
                        ctypes.byref(user),
                    )
                finally:
                    ctypes.windll.kernel32.CloseHandle(process)
                if ok:
                    value = (creation.dwHighDateTime << 32) | creation.dwLowDateTime
                    return f"win:{value}"
            except (AttributeError, OSError, ValueError):
                pass
        return None

    @classmethod
    def _same_live_process(cls, pid: object, expected_token: object) -> bool:
        if not cls._pid_alive(pid):
            return False
        if not isinstance(expected_token, str) or not expected_token:
            # Legacy receipts do not carry an identity token.  Preserve the
            # conservative live-PID behaviour for those receipts.
            return True
        return cls._pid_start_token(pid) == expected_token

    def reconcile_orphans(self) -> list[str]:
        """Seal stale local runs whose child identity is no longer alive."""

        reconciled: list[str] = []
        now = time.time()
        hostname = socket.gethostname()
        for run_dir in sorted((self.root / "runs").iterdir()):
            if not run_dir.is_dir():
                continue
            try:
                if _terminal_payload(run_dir) is not None:
                    continue
            except ValueError:
                # Never add a second terminal artifact beside corrupt or
                # conflicting external state.  It must be audited manually.
                continue
            started_path = run_dir / "started.json"
            if not started_path.is_file():
                continue
            malformed_started = False
            try:
                started = _load_json_object(started_path, label="demo started receipt")
                if (
                    started.get("schema_version") != 1
                    or started.get("run_id") != run_dir.name
                    or started.get("mode") not in {"deterministic", "agent", "agent-offline"}
                ):
                    raise ValueError("invalid started receipt schema")
                started_epoch = calendar.timegm(
                    time.strptime(started["started_at"], "%Y-%m-%dT%H:%M:%SZ")
                )
            except (KeyError, OSError, TypeError, ValueError):
                malformed_started = True
                started = {}
                try:
                    started_epoch = started_path.stat().st_mtime
                except OSError:
                    continue
            if now - started_epoch <= self.timeout:
                continue
            if started.get("hostname") not in {None, hostname}:
                continue
            child_path = run_dir / "child-started.json"
            child: dict[str, Any] = {}
            if child_path.is_file():
                try:
                    child = _load_json_object(child_path, label="demo child receipt")
                    if child.get("run_id") not in {None, run_dir.name}:
                        child = {}
                except ValueError:
                    child = {}
            has_child_identity = type(child.get("child_pid")) is int
            if has_child_identity:
                # Once a child receipt exists, the child is authoritative.
                # A live Streamlit/service process must not keep a dead child
                # marked as running forever.
                if self._same_live_process(
                    child.get("child_pid"), child.get("child_start_token")
                ):
                    continue
            elif self._same_live_process(
                started.get("service_pid"), started.get("service_start_token")
            ):
                continue
            payload = {
                "schema_version": 1,
                "run_id": started.get("run_id", run_dir.name),
                "started_at": started.get("started_at"),
                "completed_at": _utc(),
                "status": "failed",
                "returncode": None,
                "source_tree_sha256": started.get("source_tree_sha256"),
                "mode": started.get("mode"),
                "error_summary": {
                    "failure_kind": (
                        "orphaned_invalid_started_receipt"
                        if malformed_started
                        else "orphaned_after_service_restart"
                    ),
                    "exception_type": "ProcessNotAlive",
                    "exception_message": "stale demo run had no matching live child process identity",
                },
                "artifacts": {},
                "orphan_reconciled_at": _utc(),
            }
            try:
                _publish_terminal(run_dir, "failed.json", payload)
            except (FileExistsError, ValueError):
                continue
            reconciled.append(run_dir.name)
        return reconciled

    def _event(self, path: Path, *, kind: str, content: str, stream: str = "stdout") -> None:
        payload = {"timestamp": _utc(), "kind": kind, "stream": stream, "content": content.rstrip("\r\n")}
        with _EVENT_LOCK:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                handle.flush()

    @staticmethod
    def _source_tree_sha256() -> str:
        """Return the sealed included-tree hash or fail before creating a run.

        A demo receipt without a source binding cannot be used as competition
        evidence.  Do not silently substitute a sentinel such as ``unbound``.
        """

        value = (os.getenv("DKI_INCLUDED_TREE_SHA256") or "").strip()
        if _SOURCE_TREE_RE.fullmatch(value) is None:
            raise ValueError(
                "DKI_INCLUDED_TREE_SHA256 must be set to the sealed 64-hex included-tree hash"
            )
        return value.lower()

    @staticmethod
    def _prepare_text_file(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8", newline=""):
            pass

    def _drain_stream(
        self,
        stream: Any,
        log_path: Path,
        event_path: Path,
        stream_name: str,
        errors: list[tuple[str, Exception]],
    ) -> None:
        """Drain one pipe without blocking the process timeout clock."""

        try:
            with log_path.open("a", encoding="utf-8", newline="") as log:
                for line in stream:
                    text = str(line)
                    log.write(text)
                    log.flush()
                    self._event(event_path, kind="pipeline_log", content=text, stream=stream_name)
        except Exception as exc:  # captured by the main thread and sealed in failed.json
            errors.append((stream_name, exc))
        finally:
            try:
                stream.close()
            except Exception:
                pass

    @staticmethod
    def _stop_process(process: Any) -> int | None:
        """Terminate only the demo child, escalating to kill after a grace period."""

        try:
            returncode = process.poll()
        except Exception:
            returncode = None
        if returncode is not None:
            return returncode
        try:
            process.terminate()
        except Exception:
            pass
        try:
            return process.wait(timeout=5)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass
        try:
            return process.wait(timeout=5)
        except Exception:
            return getattr(process, "returncode", None)

    @staticmethod
    def _join_readers(threads: list[threading.Thread], process: Any | None) -> list[str]:
        for thread in threads:
            thread.join(timeout=5)
        alive = [thread for thread in threads if thread.is_alive()]
        if alive and process is not None:
            for stream_name in ("stdout", "stderr"):
                stream = getattr(process, stream_name, None)
                if stream is not None:
                    try:
                        stream.close()
                    except Exception:
                        pass
            for thread in alive:
                thread.join(timeout=1)
        return [thread.name for thread in threads if thread.is_alive()]

    @staticmethod
    def _exception_summary(exc: Exception) -> dict[str, str]:
        message = str(exc).replace("\r", " ").replace("\n", " ")[:2000]
        return {"exception_type": type(exc).__name__, "exception_message": message}

    def _publish_failure(
        self,
        *,
        run_id: str,
        run_dir: Path,
        started: str,
        source_tree: str,
        mode: str,
        returncode: int | None,
        failure_kind: str,
        error: Exception,
        stdout_path: Path,
        stderr_path: Path,
        event_path: Path,
    ) -> DemoRunReceipt:
        """Create the terminal failed receipt without trusting summary helpers."""

        for path in (stdout_path, stderr_path, event_path):
            if not path.exists():
                try:
                    self._prepare_text_file(path)
                except Exception:
                    # A filesystem-level failure may make preservation
                    # impossible, but it must not hide the original failure.
                    pass
        try:
            self._event(event_path, kind="run_failed", content=failure_kind, stream="service")
        except Exception:
            pass
        diagnostics: list[dict[str, str]] = []
        try:
            artifacts = self._artifact_summary(run_dir)
        except Exception as exc:
            artifacts = {}
            diagnostics.append({"stage": "artifact_summary", **self._exception_summary(exc)})
        try:
            trace = self._trace_summary(event_path, stdout_path, stderr_path)
        except Exception as exc:
            trace = {
                "datamate_mode": "not-reported",
                "agent_trace": {"event_count": 0, "recovery_event_count": 0},
                "error_summary": {"error_line_count": 0},
            }
            diagnostics.append({"stage": "trace_summary", **self._exception_summary(exc)})
        try:
            input_sha256 = self._input_data_sha256()
        except Exception as exc:
            input_sha256 = "unavailable"
            diagnostics.append({"stage": "input_data_sha256", **self._exception_summary(exc)})

        def safe_hash(path: Path, stage: str) -> str:
            try:
                return _sha256(path)
            except Exception as exc:
                diagnostics.append({"stage": stage, **self._exception_summary(exc)})
                return "unavailable"

        completed = _utc()
        error_summary = dict(trace.get("error_summary", {}))
        error_summary.update({
            "failure_kind": failure_kind,
            **self._exception_summary(error),
            "diagnostics": diagnostics,
        })
        payload = {
            "schema_version": 1,
            "run_id": run_id,
            "started_at": started,
            "completed_at": completed,
            "status": "failed",
            "returncode": returncode,
            "source_tree_sha256": source_tree,
            "mode": mode,
            "datamate_mode": trace.get("datamate_mode", "not-reported"),
            "input_data_sha256": input_sha256,
            "agent_trace": trace.get("agent_trace", {}),
            "error_summary": error_summary,
            "artifacts": artifacts,
            "stdout_sha256": safe_hash(stdout_path, "stdout_sha256"),
            "stderr_sha256": safe_hash(stderr_path, "stderr_sha256"),
        }
        _publish_terminal(run_dir, "failed.json", payload)
        return DemoRunReceipt(run_id, run_dir, "failed", returncode, started, completed, source_tree, mode)

    def run(self, *, agent: bool = True, offline: bool = True, run_id: str | None = None) -> DemoRunReceipt:
        if not _RUN_LOCK.acquire(blocking=False):
            raise RuntimeError("已有演示流水线正在运行，请等待当前 run 完成")
        try:
            # Fail before allocating a run ID or evidence directory when the
            # executable source tree has not been sealed.
            source_tree = self._source_tree_sha256()
            run_id = run_id or self.new_run_id()
            run_dir = self.root / "runs" / run_id
            run_dir.mkdir(parents=True, exist_ok=False)
            started = _utc()
            mode = "agent-offline" if agent and offline else ("agent" if agent else "deterministic")
            _write_once(run_dir / "started.json", {
                "schema_version": 1,
                "run_id": run_id,
                "started_at": started,
                "source_tree_sha256": source_tree,
                "mode": mode,
                "output_root": str(run_dir),
                "hostname": socket.gethostname(),
                "service_pid": os.getpid(),
                "service_start_token": self._pid_start_token(os.getpid()),
            })
            stdout_path = run_dir / "stdout.log"
            stderr_path = run_dir / "stderr.log"
            event_path = run_dir / "events.jsonl"
            try:
                for path in (stdout_path, stderr_path, event_path):
                    self._prepare_text_file(path)
                self._event(event_path, kind="run_started", content="demo child process starting", stream="service")
            except Exception as exc:
                # Best effort: the failure publisher will record unavailable
                # hashes if the filesystem itself prevented a log file.
                return self._publish_failure(
                    run_id=run_id, run_dir=run_dir, started=started, source_tree=source_tree,
                    mode=mode, returncode=None, failure_kind="log_initialization", error=exc,
                    stdout_path=stdout_path, stderr_path=stderr_path, event_path=event_path,
                )
            command = [self.python, "-m", "dki", "demo-all"]
            if agent:
                command.append("--agent")
            if offline:
                command.append("--offline")
            env = os.environ.copy()
            env["DKI_OUTPUT_DIR"] = str(run_dir)
            env["DKI_DEMO_SERVER_RID"] = self.server_rid
            env["PYTHONUTF8"] = "1"
            process: Any | None = None
            threads: list[threading.Thread] = []
            reader_errors: list[tuple[str, Exception]] = []
            returncode: int | None = None
            timed_out = False
            try:
                process = subprocess.Popen(
                    command,
                    cwd=Path(__file__).resolve().parents[2],
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                )
                if process.stdout is None or process.stderr is None:
                    raise RuntimeError("demo child did not expose stdout/stderr pipes")
                _write_once(run_dir / "child-started.json", {
                    "schema_version": 1,
                    "run_id": run_id,
                    "started_at": _utc(),
                    "hostname": socket.gethostname(),
                    "child_pid": getattr(process, "pid", None),
                    "child_start_token": self._pid_start_token(getattr(process, "pid", None)),
                })
                for stream, path, name in (
                    (process.stdout, stdout_path, "stdout"),
                    (process.stderr, stderr_path, "stderr"),
                ):
                    thread = threading.Thread(
                        target=self._drain_stream,
                        args=(stream, path, event_path, name, reader_errors),
                        name=f"demo-{name}-{run_id}",
                        daemon=True,
                    )
                    thread.start()
                    threads.append(thread)
                try:
                    returncode = process.wait(timeout=self.timeout)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    self._event(event_path, kind="timeout", content=f"demo timeout after {self.timeout}s")
                    returncode = self._stop_process(process)
                stuck_readers = self._join_readers(threads, process)
                if stuck_readers:
                    reader_errors.append((
                        "stream",
                        RuntimeError(f"stream readers did not stop: {', '.join(stuck_readers)}"),
                    ))
            except Exception as exc:
                if process is not None:
                    returncode = self._stop_process(process)
                    self._join_readers(threads, process)
                try:
                    self._event(event_path, kind="runtime_exception", content=self._exception_summary(exc)["exception_message"], stream="service")
                except Exception:
                    pass
                return self._publish_failure(
                    run_id=run_id, run_dir=run_dir, started=started, source_tree=source_tree,
                    mode=mode, returncode=returncode, failure_kind="process_launch_or_runtime", error=exc,
                    stdout_path=stdout_path, stderr_path=stderr_path, event_path=event_path,
                )

            if timed_out:
                return self._publish_failure(
                    run_id=run_id, run_dir=run_dir, started=started, source_tree=source_tree,
                    mode=mode, returncode=returncode, failure_kind="timeout",
                    error=TimeoutError(f"demo timeout after {self.timeout}s"),
                    stdout_path=stdout_path, stderr_path=stderr_path, event_path=event_path,
                )
            if reader_errors:
                stream_name, stream_error = reader_errors[0]
                return self._publish_failure(
                    run_id=run_id, run_dir=run_dir, started=started, source_tree=source_tree,
                    mode=mode, returncode=returncode, failure_kind=f"{stream_name}_read",
                    error=stream_error, stdout_path=stdout_path, stderr_path=stderr_path,
                    event_path=event_path,
                )
            if returncode != 0:
                return self._publish_failure(
                    run_id=run_id, run_dir=run_dir, started=started, source_tree=source_tree,
                    mode=mode, returncode=returncode, failure_kind="subprocess_exit",
                    error=RuntimeError(f"demo-all exited with return code {returncode}"),
                    stdout_path=stdout_path, stderr_path=stderr_path, event_path=event_path,
                )

            try:
                artifacts = self._artifact_summary(run_dir)
                runtime = self._runtime_summary(
                    run_dir,
                    source_tree,
                    expected_mode=mode,
                    artifacts=artifacts,
                )
                log_trace = self._trace_summary(event_path, stdout_path, stderr_path)
                runtime_values = runtime["runtime"]
                trace = {
                    "datamate_mode": runtime_values["datamate_mode"],
                    "agent_trace": runtime["agent_trace"],
                    "error_summary": log_trace["error_summary"],
                }
                input_sha256 = runtime["input_snapshot"]["sha256"]
                stdout_sha256 = _sha256(stdout_path)
                stderr_sha256 = _sha256(stderr_path)
            except Exception as exc:
                return self._publish_failure(
                    run_id=run_id, run_dir=run_dir, started=started, source_tree=source_tree,
                    mode=mode, returncode=returncode, failure_kind="summary", error=exc,
                    stdout_path=stdout_path, stderr_path=stderr_path, event_path=event_path,
                )
            completed = _utc()
            payload = {
                "schema_version": 1,
                "run_id": run_id,
                "started_at": started,
                "completed_at": completed,
                "status": "completed",
                "returncode": returncode,
                "source_tree_sha256": source_tree,
                "mode": mode,
                "datamate_mode": trace["datamate_mode"],
                "input_data_sha256": input_sha256,
                "agent_trace": trace["agent_trace"],
                "error_summary": trace["error_summary"],
                "runtime": runtime_values,
                "runtime_receipt_sha256": artifacts["demo_runtime"]["sha256"],
                "artifacts": artifacts,
                "stdout_sha256": stdout_sha256,
                "stderr_sha256": stderr_sha256,
            }
            try:
                _publish_terminal(run_dir, "completed.json", payload)
            except Exception as exc:
                # A successful canonical terminal claim is authoritative even
                # if an alias-level filesystem fault occurred.  Never publish
                # failed.json beside it.
                try:
                    terminal = _terminal_payload(run_dir)
                except ValueError:
                    terminal = None
                if terminal is not None and terminal.get("status") == "completed":
                    return DemoRunReceipt(
                        run_id, run_dir, "completed", returncode, started,
                        completed, source_tree, mode,
                    )
                return self._publish_failure(
                    run_id=run_id, run_dir=run_dir, started=started, source_tree=source_tree,
                    mode=mode, returncode=returncode, failure_kind="completed_receipt_publish", error=exc,
                    stdout_path=stdout_path, stderr_path=stderr_path, event_path=event_path,
                )
            return DemoRunReceipt(run_id, run_dir, "completed", returncode, started, completed, source_tree, mode)
        finally:
            _RUN_LOCK.release()

    def _input_data_sha256(self) -> str:
        """Hash only the public synthetic input tree; never read private holdouts."""
        input_root = Path(__file__).resolve().parents[2] / "data" / "raw"
        if not input_root.is_dir():
            return "missing"
        digest = hashlib.sha256()
        for path in sorted(item for item in input_root.rglob("*") if item.is_file()):
            digest.update(str(path.relative_to(input_root)).replace("\\", "/").encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
        return digest.hexdigest()

    @staticmethod
    def _trace_summary(event_path: Path, stdout_path: Path, stderr_path: Path | None = None) -> dict[str, Any]:
        lines = stdout_path.read_text(encoding="utf-8", errors="replace").splitlines() if stdout_path.is_file() else []
        if stderr_path is not None and stderr_path.is_file():
            lines.extend(stderr_path.read_text(encoding="utf-8", errors="replace").splitlines())
        event_count = 0
        recovery_count = 0
        if event_path.is_file():
            event_text = event_path.read_text(encoding="utf-8", errors="replace")
            event_count = len(event_text.splitlines())
            recovery_count = sum(1 for line in event_text.splitlines() if "recovery" in line.casefold())
        lowered = "\n".join(lines).casefold()
        if "compat" in lowered or "shim" in lowered:
            datamate_mode = "compat-shim"
        elif "datamate" in lowered:
            datamate_mode = "real-datamate-or-runtime"
        else:
            datamate_mode = "not-reported"
        # Structured success payloads often contain ``errors: []`` and
        # ``error: ''``.  They are not failures, so only count explicit
        # severity/exception markers or a terminal FAILED/ok=false status.
        # Keep severity matching case-sensitive so JSON fields named
        # ``error``/``errors`` are not mistaken for log-level records.
        severity_pattern = re.compile(r"\b(?:ERROR|CRITICAL)\b")
        exception_pattern = re.compile(
            r"Traceback \(most recent call last\)|\b[A-Z][A-Za-z_]*Error\b"
        )
        terminal_pattern = re.compile(
            r"\bstatus\s*[:=]\s*['\"]?FAILED\b|\bok\s*[:=]\s*False\b",
            re.IGNORECASE,
        )
        error_tokens = sum(
            1
            for line in lines
            if severity_pattern.search(line)
            or exception_pattern.search(line)
            or terminal_pattern.search(line)
        )
        return {
            "datamate_mode": datamate_mode,
            "agent_trace": {"event_count": event_count, "recovery_event_count": recovery_count},
            "error_summary": {"error_line_count": error_tokens},
        }

    @staticmethod
    def _artifact_summary(run_dir: Path) -> dict[str, Any]:
        from dki.analysis.database import database_matches_manifest
        from dki.analysis.nl2sql_sqlite import connect_readonly
        from dki.kg.store import KnowledgeGraph
        from dki.pipeline import artifacts as artifact_contract

        run_dir = Path(run_dir).resolve()
        candidates = {
            "task1_manifest": run_dir / "processed" / "manifest.json",
            "kg": run_dir / "kg" / "medical_kg.json",
            "kg_provenance": run_dir / "kg" / "medical_kg.provenance.json",
            "database": run_dir / "insight.db",
            "database_provenance": run_dir / "insight.provenance.json",
            "bi_report": run_dir / "reports" / "insight_report.html",
            "demo_runtime": run_dir / "demo_runtime.json",
        }
        result: dict[str, Any] = {}
        missing: list[str] = []
        for name, path in candidates.items():
            if path.is_file():
                if path.is_symlink():
                    raise ValueError(f"completed demo artifact must not be a symlink: {name}")
                try:
                    path.resolve().relative_to(run_dir)
                except ValueError as exc:
                    raise ValueError(
                        f"completed demo artifact escapes the isolated run: {name}"
                    ) from exc
                result[name] = {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
            else:
                missing.append(name)
        if missing:
            raise FileNotFoundError(
                "completed demo is missing required artifacts: " + ", ".join(missing)
            )

        def resolve_binding(raw: object, *, label: str) -> Path:
            if not isinstance(raw, str) or not raw:
                raise ValueError(f"{label} path is invalid")
            path = Path(raw)
            if not path.is_absolute():
                path = artifact_contract.PROJECT_ROOT / path
            return path.resolve()

        def require_run_scoped(path: Path, *, label: str) -> None:
            try:
                path.resolve().relative_to(run_dir)
            except ValueError as exc:
                raise ValueError(f"{label} escapes the isolated demo run") from exc

        manifest_path = candidates["task1_manifest"]
        manifest = _load_json_object(manifest_path, label="Task1 manifest")
        verified_manifest = artifact_contract.load_task1_manifest(
            manifest_path, verify_hashes=True
        )
        if dict(verified_manifest) != manifest:
            raise ValueError("Task1 manifest changed while it was being validated")
        clean = manifest.get("clean_text")
        tables = manifest.get("structured_tables")
        task_report_binding = manifest.get("task_report")
        if (
            not isinstance(clean, dict)
            or not isinstance(clean.get("files"), list)
            or clean.get("count") != len(clean.get("files", []))
            or not isinstance(tables, dict)
            or not isinstance(task_report_binding, dict)
        ):
            raise ValueError("Task1 manifest lacks required artifact bindings")
        scoped_bindings = [*clean["files"], *tables.values(), task_report_binding]
        clean_dir = resolve_binding(clean.get("directory"), label="Task1 clean directory")
        require_run_scoped(clean_dir, label="Task1 clean directory")
        for index, binding in enumerate(scoped_bindings):
            if not isinstance(binding, dict):
                raise ValueError(f"Task1 artifact binding {index} is invalid")
            bound = resolve_binding(binding.get("path"), label=f"Task1 artifact {index}")
            require_run_scoped(bound, label=f"Task1 artifact {index}")
            if not bound.is_file() or binding.get("sha256") != _sha256(bound):
                raise ValueError(f"Task1 artifact {index} hash binding failed")
            if "bytes" in binding and binding.get("bytes") != bound.stat().st_size:
                raise ValueError(f"Task1 artifact {index} size binding failed")
        for table_name, binding in tables.items():
            table_path = resolve_binding(
                binding.get("path"), label=f"Task1 table {table_name}"
            )
            try:
                with table_path.open(encoding="utf-8", newline="") as handle:
                    reader = csv.reader(handle)
                    columns = next(reader, [])
                    row_count = sum(1 for _ in reader)
            except (OSError, UnicodeError, csv.Error) as exc:
                raise ValueError(f"Task1 table {table_name} is unreadable") from exc
            if binding.get("columns") != columns or binding.get("row_count") != row_count:
                raise ValueError(f"Task1 table {table_name} schema/row binding failed")
        task_report_path = resolve_binding(
            task_report_binding.get("path"), label="Task1 task report"
        )
        task_report = _load_json_object(task_report_path, label="Task1 task report")
        if (
            task_report.get("status") != "COMPLETED"
            or task_report.get("task_id") != manifest.get("task_id")
        ):
            raise ValueError("Task1 task report is not the completed manifest task")

        kg_path = candidates["kg"]
        kg_provenance_path = candidates["kg_provenance"]
        kg_provenance = _load_json_object(
            kg_provenance_path, label="Task2 KG provenance"
        )
        artifact_contract.validate_task2_artifact(
            kg_path, task1_path=manifest_path
        )
        if (
            kg_provenance.get("task1_task_id") != manifest.get("task_id")
            or resolve_binding(
                kg_provenance.get("task1_manifest_path"), label="Task2 Task1 binding"
            )
            != manifest_path.resolve()
            or resolve_binding(kg_provenance.get("kg_path"), label="Task2 KG binding")
            != kg_path.resolve()
        ):
            raise ValueError("Task2 provenance path/task binding failed")
        raw_kg = _load_json_object(kg_path, label="Task2 knowledge graph")
        if not isinstance(raw_kg.get("entities"), dict) or not isinstance(
            raw_kg.get("triples"), list
        ):
            raise ValueError("Task2 knowledge graph has an invalid public schema")
        try:
            graph = KnowledgeGraph.load(kg_path)
            # Force every triple through the public accessor after load.
            if len(graph.triples) != len(raw_kg["triples"]):
                raise ValueError("Task2 knowledge graph lost triples during loading")
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Task2 knowledge graph cannot be loaded") from exc

        db_path = candidates["database"]
        db_provenance_path = candidates["database_provenance"]
        db_provenance = _load_json_object(
            db_provenance_path, label="Task3 database provenance"
        )
        if not database_matches_manifest(db_path, manifest_path, kg_path):
            raise ValueError("Task3 database provenance/hash binding failed")
        if (
            db_provenance.get("schema_version") != 1
            or resolve_binding(
                db_provenance.get("task1_manifest_path"), label="Task3 Task1 binding"
            )
            != manifest_path.resolve()
            or resolve_binding(
                db_provenance.get("task2_kg_path"), label="Task3 KG binding"
            )
            != kg_path.resolve()
            or resolve_binding(
                db_provenance.get("database_path"), label="Task3 database binding"
            )
            != db_path.resolve()
        ):
            raise ValueError("Task3 database provenance path binding failed")
        connection = connect_readonly(db_path)
        try:
            found_tables = {
                str(row[0])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
            }
            required_tables = {"patients", "visits", "kg_entities", "kg_triples"}
            if not required_tables.issubset(found_tables):
                raise ValueError("Task3 database lacks required public tables")
            for table in sorted(required_tables):
                connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()
        finally:
            connection.close()

        try:
            html = candidates["bi_report"].read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise ValueError("BI report is not valid UTF-8 HTML") from exc
        lowered_html = html.casefold()
        if (
            len(html) < 128
            or "<!doctype html" not in lowered_html
            or "<html" not in lowered_html
            or "echarts" not in lowered_html
            or "const charts" not in lowered_html
        ):
            raise ValueError("BI report does not contain the required ECharts document")
        for name, path in candidates.items():
            if (
                not path.is_file()
                or path.stat().st_size != result[name]["bytes"]
                or _sha256(path) != result[name]["sha256"]
            ):
                raise ValueError(
                    f"completed demo artifact changed during validation: {name}"
                )
        return result

    @staticmethod
    def _input_snapshot_from_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
        source = manifest.get("source")
        rows = source.get("raw_text_files") if isinstance(source, dict) else None
        if not isinstance(rows, list) or not rows:
            raise ValueError("Task1 manifest lacks a non-empty raw input snapshot")
        canonical: list[tuple[str, int, str]] = []
        seen: set[str] = set()
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValueError(f"Task1 input snapshot row {index} is invalid")
            raw_path = row.get("path")
            size = row.get("bytes")
            sha256 = row.get("sha256")
            normalized = raw_path.replace("\\", "/") if isinstance(raw_path, str) else ""
            if (
                not normalized
                or normalized in seen
                or type(size) is not int
                or size < 0
                or not isinstance(sha256, str)
                or _SOURCE_TREE_RE.fullmatch(sha256) is None
            ):
                raise ValueError(f"Task1 input snapshot row {index} is not hash-bound")
            seen.add(normalized)
            canonical.append((normalized, size, sha256.lower()))
        digest = hashlib.sha256()
        for path, size, sha256 in sorted(canonical):
            digest.update(path.encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(size).encode("ascii"))
            digest.update(b"\0")
            digest.update(sha256.encode("ascii"))
            digest.update(b"\n")
        return {
            "algorithm": "sha256 of sorted path NUL size NUL file_sha256 LF",
            "n_files": len(canonical),
            "sha256": digest.hexdigest(),
        }

    @staticmethod
    def _runtime_summary(
        run_dir: Path,
        source_tree_sha256: str,
        *,
        expected_mode: str | None = None,
        artifacts: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        path = Path(run_dir) / "demo_runtime.json"
        payload = _load_json_object(path, label="demo runtime receipt")
        runtime = payload.get("runtime")
        input_snapshot = payload.get("input_snapshot")
        agent_trace = payload.get("agent_trace")
        task1_binding = payload.get("task1_manifest")
        task_report_binding = payload.get("task1_report")
        runtime_artifacts = payload.get("artifacts")
        pipeline = payload.get("pipeline")
        privacy = payload.get("privacy")
        task1_manifest = Path(run_dir) / "processed" / "manifest.json"
        manifest = _load_json_object(task1_manifest, label="Task1 manifest")
        expected_snapshot = DemoRunService._input_snapshot_from_manifest(manifest)
        manifest_report = manifest.get("task_report")
        expected_artifact_paths = {
            "kg": Path(run_dir) / "kg" / "medical_kg.json",
            "kg_provenance": Path(run_dir) / "kg" / "medical_kg.provenance.json",
            "database": Path(run_dir) / "insight.db",
            "database_provenance": Path(run_dir) / "insight.provenance.json",
            "bi_report": Path(run_dir) / "reports" / "insight_report.html",
        }

        def receipt_path(raw: object) -> Path:
            candidate = Path(str(raw))
            if not candidate.is_absolute():
                candidate = Path(__file__).resolve().parents[2] / candidate
            return candidate.resolve()

        def valid_binding(binding: object, target: Path) -> bool:
            if not isinstance(binding, dict):
                return False
            try:
                raw_path = binding.get("path")
                if not isinstance(raw_path, str):
                    return False
                return (
                    receipt_path(raw_path) == target.resolve()
                    and binding.get("bytes") == target.stat().st_size
                    and binding.get("sha256") == _sha256(target)
                )
            except (OSError, ValueError):
                return False

        def valid_core_agent(value: object) -> bool:
            return (
                isinstance(value, str)
                and value.startswith("nexent.")
                and value.endswith(".CoreAgent")
            )

        mode_valid = isinstance(runtime, dict) and isinstance(agent_trace, dict)
        if expected_mode is not None:
            if not mode_valid:
                mode_valid = False
            elif expected_mode == "deterministic":
                mode_valid = (
                    runtime.get("agent_enabled") is False
                    and runtime.get("core_agent") is None
                    and runtime.get("model_backend") is None
                    and agent_trace.get("tool_calls") == []
                )
            elif expected_mode == "agent-offline":
                mode_valid = (
                    runtime.get("agent_enabled") is True
                    and runtime.get("offline") is True
                    and valid_core_agent(runtime.get("core_agent"))
                    and runtime.get("model_backend") == "ScriptedModel"
                    and bool(agent_trace.get("tool_calls"))
                )
            elif expected_mode == "agent":
                mode_valid = (
                    runtime.get("agent_enabled") is True
                    and runtime.get("offline") is False
                    and valid_core_agent(runtime.get("core_agent"))
                    and runtime.get("model_backend") == "OpenAIModel"
                )
            else:
                mode_valid = False
        if (
            payload.get("schema_version") != 1
            or payload.get("kind") != "dki_demo_runtime_receipt"
            or payload.get("status") != "completed"
            or payload.get("source_tree_sha256") != source_tree_sha256
            or payload.get("evidence_eligible") is not True
            or not isinstance(runtime, dict)
            or runtime.get("datamate_mode") not in {"real-datamate", "compat-shim"}
            or type(runtime.get("datamate_available")) is not bool
            or (runtime.get("datamate_mode") == "real-datamate")
            is not runtime.get("datamate_available")
            or type(runtime.get("agent_enabled")) is not bool
            or type(runtime.get("offline")) is not bool
            or not isinstance(agent_trace, dict)
            or not isinstance(agent_trace.get("tool_calls"), list)
            or input_snapshot != expected_snapshot
            or not isinstance(task1_binding, dict)
            or receipt_path(task1_binding.get("path", "")) != task1_manifest.resolve()
            or task1_binding.get("sha256") != _sha256(task1_manifest)
            or task1_binding.get("bytes") != task1_manifest.stat().st_size
            or not isinstance(manifest_report, dict)
            or not valid_binding(
                task_report_binding,
                receipt_path(manifest_report.get("path", "")),
            )
            or task_report_binding.get("sha256") != manifest_report.get("sha256")
            or task_report_binding.get("bytes") != manifest_report.get("bytes")
            or not isinstance(runtime_artifacts, dict)
            or any(
                not valid_binding(runtime_artifacts.get(name), target)
                for name, target in expected_artifact_paths.items()
            )
            or (
                artifacts is not None
                and any(
                    artifacts.get(name, {}).get("sha256") != _sha256(target)
                    for name, target in expected_artifact_paths.items()
                )
            )
            or not isinstance(pipeline, dict)
            or pipeline.get("status") != "COMPLETED"
            or pipeline.get("n_failed") not in {0, None}
            or not mode_valid
            or not isinstance(privacy, dict)
            or privacy.get("contains_secrets") is not False
            or privacy.get("contains_private_blind_data") is not False
            or privacy.get("contains_full_user_input") is not False
        ):
            raise ValueError("demo runtime receipt does not satisfy the public schema")
        return payload

    @staticmethod
    def read_receipt(run_dir: Path) -> dict[str, Any] | None:
        try:
            run_dir = Path(run_dir).resolve()
            terminal = _terminal_payload(run_dir)
            if terminal is not None:
                result = dict(terminal)
                result["_service_verified"] = False
                if terminal.get("status") != "completed":
                    return result
                source_tree = terminal.get("source_tree_sha256")
                mode = terminal.get("mode")
                if (
                    terminal.get("run_id") != run_dir.name
                    or _SOURCE_TREE_RE.fullmatch(str(source_tree)) is None
                    or mode not in {"deterministic", "agent", "agent-offline"}
                ):
                    return None
                artifacts = DemoRunService._artifact_summary(run_dir)
                DemoRunService._runtime_summary(
                    run_dir,
                    str(source_tree),
                    expected_mode=str(mode),
                    artifacts=artifacts,
                )
                recorded = terminal.get("artifacts")
                if not isinstance(recorded, dict) or any(
                    not isinstance(recorded.get(name), dict)
                    or recorded[name].get("sha256") != binding["sha256"]
                    or recorded[name].get("bytes") != binding["bytes"]
                    for name, binding in artifacts.items()
                ):
                    return None
                if terminal.get("runtime_receipt_sha256") != artifacts["demo_runtime"]["sha256"]:
                    return None
                result["_service_verified"] = True
                return result
            path = run_dir / "started.json"
            if not path.is_file():
                return None
            result = _load_json_object(path, label="demo started receipt")
            result["_service_verified"] = False
            return result
        except (OSError, TypeError, ValueError):
            return None

    @staticmethod
    def tail_events(run_dir: Path, limit: int = 120) -> list[dict[str, Any]]:
        path = Path(run_dir) / "events.jsonl"
        if not path.is_file():
            return []
        rows: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()[-limit:]:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return rows


def iter_run_artifacts(run_dir: Path) -> Iterator[tuple[str, Path]]:
    """Yield only public demo artifacts; private logs stay behind the UI."""

    for relative in (
        "processed/manifest.json",
        "kg/medical_kg.json",
        "kg/medical_kg.provenance.json",
        "insight.db",
        "insight.provenance.json",
        "reports/insight_report.html",
        "demo_runtime.json",
    ):
        path = Path(run_dir) / relative
        if path.is_file():
            yield relative, path
