"""下载并规范化 CBLUE 真实医疗基准（CMeEE-V2 命名实体 / CMeIE-V2 关系抽取）。

来源：HuggingFace 社区镜像（hf-mirror，国内可达），用 hf_hub_download 逐文件抓取 +
有界重试（比 snapshot_download 稳）。规范化为统一 jsonl：
  data/cblue/CMeEE-V2/{train,dev}.jsonl   每行 {"text","entities":[{"start","end","type"}]}
  data/cblue/CMeIE-V2/{train,dev}.jsonl   每行 {"text","spo_list":[{"subject","predicate","object"}]}

许可与合规：CBLUE 供学术研究使用；本仓库仅本地下载训练/评测，**不重分发原始语料**，
提交包只含数据卡 + 复现脚本（见 docs/cblue_datacard.md）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
DATA_DIR = ROOT / "data" / "cblue"


_MIRROR = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")


def _dl(repo_id: str, filename: str, retries: int = 6):
    """直连 hf-mirror 的 dataset resolve URL 流式下载（比 hf_hub_download 的元数据解析稳）。"""
    import requests
    url = f"{_MIRROR}/datasets/{repo_id}/resolve/main/{filename}"
    cache = ROOT / "data" / "cblue" / "_cache" / repo_id.replace("/", "__") / filename.replace("/", "__")
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.is_file() and cache.stat().st_size > 0:
        return str(cache)
    last = None
    for i in range(retries):
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with cache.open("wb") as fh:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        fh.write(chunk)
            if cache.stat().st_size > 0:
                return str(cache)
        except Exception as exc:
            last = exc
            time.sleep(3 * (i + 1))
    raise RuntimeError(f"download failed {url}: {str(last)[:160]}")


def _read_parquet(path: str):
    import pandas as pd
    return pd.read_parquet(path).to_dict("records")


def _as_list(v):
    """把 pandas/numpy 的 list 列（ndarray）安全转为 Python list。"""
    if v is None:
        return []
    if isinstance(v, str):
        try:
            return json.loads(v)
        except Exception:
            return []
    try:
        import numpy as np
        if isinstance(v, np.ndarray):
            return list(v)
    except Exception:
        pass
    if isinstance(v, (list, tuple)):
        return list(v)
    return []


def _norm_cmeee_row(row: dict) -> dict:
    """xusenlin/cmeee 行 -> {text, entities:[{start,end,type,entity}]}。

    实测字段：text + entities(ndarray of {entity,start_offset,end_offset,label})。
    start_offset/end_offset 已是**半开区间** [start,end)（end-start==len），直接沿用。
    """
    text = row.get("text") or row.get("sentence") or ""
    out = []
    for e in _as_list(row.get("entities")):
        if not isinstance(e, dict):
            continue
        s = e.get("start_offset", e.get("start_idx", e.get("start")))
        t = e.get("end_offset", e.get("end_idx", e.get("end")))
        typ = e.get("label") or e.get("type") or e.get("entity_type")
        if s is None or t is None or typ is None:
            continue
        s, t = int(s), int(t)
        ent = e.get("entity") or e.get("text") or text[s:t]
        out.append({"start": s, "end": t, "type": str(typ), "entity": ent})
    return {"text": text, "entities": out}


def _norm_cmeie_row(row: dict) -> dict:
    """Normalize CMeIE without discarding relation schema/type metadata."""
    text = row.get("text") or row.get("sentence") or ""
    spo = _as_list(row.get("spo_list") or row.get("spos") or row.get("relations"))
    out = []
    for relation in spo:
        if not isinstance(relation, dict):
            continue
        subject = relation.get("subject") or (relation.get("h") or {}).get("name")
        predicate = relation.get("predicate") or relation.get("relation") or relation.get("type")
        obj_raw = relation.get("object")
        object_type_raw = relation.get("object_type")
        if isinstance(obj_raw, dict):
            obj = obj_raw.get("@value") or obj_raw.get("name") or next(iter(obj_raw.values()), "")
        else:
            obj = obj_raw
        if isinstance(object_type_raw, dict):
            object_type = object_type_raw.get("@value") or object_type_raw.get("name") \
                or next(iter(object_type_raw.values()), "")
        else:
            object_type = object_type_raw
        subject_type = relation.get("subject_type") or relation.get("h_type") or ""
        if subject and predicate and obj:
            normalized = {
                "subject": str(subject), "predicate": str(predicate), "object": str(obj),
            }
            if subject_type:
                normalized["subject_type"] = str(subject_type)
            if object_type:
                normalized["object_type"] = str(object_type)
            # Combined predicates carry an additional object slot in official CMeIE.
            if isinstance(obj_raw, dict):
                extras = {str(key): value for key, value in obj_raw.items() if key not in {"@value", "name"}}
                if extras:
                    normalized["object_fields"] = extras
            out.append(normalized)
    return {"text": text, "spo_list": out}


def _write_jsonl(path: Path, rows: list, norm):
    path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            rec = norm(r)
            if rec.get("text"):
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept += 1
    return kept


def download_cmeee() -> dict:
    # xusenlin/cmeee: train/validation/test parquet（validation 带金标）
    repo = "xusenlin/cmeee"
    train_pq = _dl(repo, "data/train-00000-of-00001-5818996d554dca6b.parquet")
    dev_pq = _dl(repo, "data/validation-00000-of-00001-dca907a321018965.parquet")
    out = DATA_DIR / "CMeEE-V2"
    n_train = _write_jsonl(out / "train.jsonl", _read_parquet(train_pq), _norm_cmeee_row)
    n_dev = _write_jsonl(out / "dev.jsonl", _read_parquet(dev_pq), _norm_cmeee_row)
    return {"repo": repo, "train": n_train, "dev": n_dev, "dir": str(out)}


def download_cmeie() -> dict:
    repo = "wubingheng/CMeIE-V2-NER"
    train = _dl(repo, "CMeIE-V2_train.jsonl")
    dev = _dl(repo, "CMeIE-V2_dev.jsonl")
    out = DATA_DIR / "CMeIE-V2"

    def _load_jsonl(p):
        return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]

    n_train = _write_jsonl(out / "train.jsonl", _load_jsonl(train), _norm_cmeie_row)
    n_dev = _write_jsonl(out / "dev.jsonl", _load_jsonl(dev), _norm_cmeie_row)
    return {"repo": repo, "train": n_train, "dev": n_dev, "dir": str(out)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["cmeee", "cmeie", "all"], default="all")
    args = ap.parse_args()
    result = {}
    if args.task in ("cmeee", "all"):
        result["cmeee"] = download_cmeee()
    if args.task in ("cmeie", "all"):
        try:
            result["cmeie"] = download_cmeie()
        except Exception as exc:
            result["cmeie"] = {"error": str(exc)[:200]}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
