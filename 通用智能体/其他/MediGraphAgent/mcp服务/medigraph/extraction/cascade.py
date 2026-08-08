"""Shared resources and merge logic for the confidence-routed cascade."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from config.settings import get_extraction_config
from medigraph.extraction.entity_linker import EntityLinker
from medigraph.extraction.fast_path import FastSpanRelationExtractor
from medigraph.schema.normalize import canonical_key


@lru_cache(maxsize=4)
def load_fast_extractor(
    artifact_path: str = "",
    calibration_path: str = "",
) -> FastSpanRelationExtractor | None:
    config = get_extraction_config()
    artifact = Path(artifact_path) if artifact_path else config.fast_artifact
    calibration = Path(calibration_path) if calibration_path else config.calibration_artifact
    if not artifact.exists():
        return None
    return FastSpanRelationExtractor.load(
        artifact,
        calibration_path=calibration if calibration.exists() else None,
    )


@lru_cache(maxsize=2)
def load_neural_extractor(model_dir: str = ""):
    """Load the trained neural GPLinker L1 if its checkpoint and the ML stack are
    present; otherwise return ``None`` so callers fall back to the lexicon path.

    Kept lazy so importing the cascade never requires torch.
    """
    from config.settings import get_extraction_config

    target = Path(model_dir) if model_dir else get_extraction_config().neural_model_dir
    try:
        from medigraph.extraction.neural_gplinker import NeuralGPLinkerExtractor
    except Exception:  # noqa: BLE001  (torch/transformers absent)
        return None
    if not NeuralGPLinkerExtractor.available(target):
        return None
    try:
        return NeuralGPLinkerExtractor(target)
    except Exception:  # noqa: BLE001  (no GPU / corrupt checkpoint)
        return None


@lru_cache(maxsize=4)
def load_entity_linker(path: str = "") -> EntityLinker | None:
    config = get_extraction_config()
    artifact = Path(path) if path else config.linker_artifact
    if not artifact.exists():
        return None
    return EntityLinker.load(artifact)


def merge_entities(primary: list[dict], secondary: list[dict]) -> list[dict]:
    """Confidence-aware union retaining provenance from both extraction levels."""
    merged: dict[tuple[str, str], dict] = {}
    for item in [*primary, *secondary]:
        key = (str(item.get("type", "")), canonical_key(str(item.get("name", ""))))
        if not key[1]:
            continue
        value = dict(item)
        if key not in merged:
            merged[key] = value
            continue
        old = merged[key]
        sources = set(old.get("cascade_sources", [old.get("extractor", "unknown")]))
        sources.update(value.get("cascade_sources", [value.get("extractor", "unknown")]))
        if float(value.get("confidence", 0.0)) > float(old.get("confidence", 0.0)):
            merged[key] = value
        merged[key]["cascade_sources"] = sorted(str(source) for source in sources if source)
    return sorted(
        merged.values(),
        key=lambda item: (
            int(item.get("start", 10**9)),
            str(item.get("type", "")),
            str(item.get("name", "")),
        ),
    )


def merge_triples(primary: list[dict], secondary: list[dict]) -> list[dict]:
    merged: dict[tuple[str, str, str], dict] = {}
    for item in [*primary, *secondary]:
        key = (
            canonical_key(str(item.get("head", ""))),
            str(item.get("relation", "")),
            canonical_key(str(item.get("tail", ""))),
        )
        if not all(key):
            continue
        value = dict(item)
        if key not in merged:
            merged[key] = value
            continue
        old = merged[key]
        sources = set(old.get("cascade_sources", [old.get("extractor", "unknown")]))
        sources.update(value.get("cascade_sources", [value.get("extractor", "unknown")]))
        if float(value.get("confidence", 0.0)) > float(old.get("confidence", 0.0)):
            merged[key] = value
        merged[key]["cascade_sources"] = sorted(str(source) for source in sources if source)
    return sorted(
        merged.values(),
        key=lambda item: (-float(item.get("confidence", 0.0)), item["head"], item["relation"], item["tail"]),
    )
