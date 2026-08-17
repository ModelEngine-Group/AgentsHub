"""
任务一输出数据集后处理模块。
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable


def _psql(sql: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "docker",
            "exec",
            "-i",
            "datamate-database",
            "psql",
            "-U",
            "postgres",
            "-d",
            "datamate",
            "-t",
            "-A",
            "-F",
            "\t",
        ],
        input=sql,
        capture_output=True,
        text=True,
    )


def postprocess_output(
    dest_dataset_id: str,
    task_id: str,
    dataset_volume: str,
    sudo_command: Callable[[list[str]], subprocess.CompletedProcess[str]],
) -> dict:
    """清理 DataMate 输出产物并识别清洗后文件名。

    Returns {removed_phantoms, removed_dups, renamed, real_count}.
    """
    res = {"removed_phantoms": 0, "removed_dups": 0, "renamed": [], "real_count": 0}
    if not dest_dataset_id:
        res["error"] = "缺少 dest_dataset_id"
        return res
    try:
        cnt = _psql(
            f"SELECT count(*) FROM t_clean_result "
            f"WHERE instance_id='{task_id}' AND dest_size=0;"
        )
        res["removed_phantoms"] = int((cnt.stdout or "0").strip() or 0)
        _psql(f"DELETE FROM t_clean_result WHERE instance_id='{task_id}' AND dest_size=0;")
        _psql(
            f"DELETE FROM t_dm_dataset_files "
            f"WHERE dataset_id='{dest_dataset_id}' AND file_size=0;"
        )

        q0 = _psql(
            f"SELECT id, file_name, file_size FROM t_dm_dataset_files "
            f"WHERE dataset_id='{dest_dataset_id}' AND file_size>0 ORDER BY id;"
        )
        seen = {}
        for line in [l for l in (q0.stdout or "").strip().splitlines() if l.count("\t") == 2]:
            fid, fname, _sz = line.split("\t", 2)
            md5 = sudo_command(["md5sum", f"{dataset_volume}/{dest_dataset_id}/{fname}"]).stdout.split()[0:1]
            key = md5[0] if md5 else fid
            if key in seen:
                sudo_command(["rm", "-f", f"{dataset_volume}/{dest_dataset_id}/{fname}"])
                _psql(f"DELETE FROM t_dm_dataset_files WHERE id='{fid}';")
                _psql(
                    f"DELETE FROM t_clean_result "
                    f"WHERE instance_id='{task_id}' AND dest_file_id='{fid}';"
                )
                res["removed_dups"] += 1
            else:
                seen[key] = fname

        q = _psql(
            f"SELECT id, file_name FROM t_dm_dataset_files "
            f"WHERE dataset_id='{dest_dataset_id}' AND file_size>0;"
        )
        rows = [l for l in (q.stdout or "").strip().splitlines() if "\t" in l]
        res["real_count"] = len(rows)
        for line in rows:
            fid, fname = line.split("\t", 1)
            if ".cleaned." in fname:
                continue
            base, dot, ext = fname.rpartition(".")
            new_name = f"{base}.cleaned.{ext}" if dot else f"{fname}.cleaned"
            sudo_command(
                [
                    "mv",
                    f"{dataset_volume}/{dest_dataset_id}/{fname}",
                    f"{dataset_volume}/{dest_dataset_id}/{new_name}",
                ]
            )
            new_path = f"/dataset/{dest_dataset_id}/{new_name}"
            _psql(
                f"UPDATE t_dm_dataset_files SET file_name='{new_name}', "
                f"file_path='{new_path}' WHERE id='{fid}';"
            )
            _psql(
                f"UPDATE t_clean_result SET dest_name='{new_name}' "
                f"WHERE instance_id='{task_id}' AND dest_name='{fname}';"
            )
            res["renamed"].append([fname, new_name])

        _psql(f"UPDATE t_clean_task SET file_count={res['real_count']} WHERE id='{task_id}';")
    except Exception as exc:
        res["error"] = str(exc)
    return res
