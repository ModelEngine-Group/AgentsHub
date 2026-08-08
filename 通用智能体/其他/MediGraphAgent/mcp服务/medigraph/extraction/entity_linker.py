"""Entity normalization/linking with stable IDs and auditable match scores."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable

from medigraph.schema.normalize import ALIASES, canonical_key, canonical_name


def stable_entity_id(name: str, entity_type: str = "") -> str:
    """Generate a deterministic local canonical ID.

    External identifiers can be supplied in a knowledge-base artifact.  When
    none is available, this stable ID supports repeatable cross-document merges
    without pretending to be an ICD identifier.
    """
    payload = f"{entity_type}:{canonical_key(name)}".encode("utf-8")
    return f"MEDIGRAPH:{hashlib.sha256(payload).hexdigest()[:16]}"


@dataclass
class LinkEntry:
    canonical_id: str
    name: str
    type: str = ""
    aliases: list[str] = field(default_factory=list)
    source: str = "MediGraph"

    def surfaces(self) -> list[str]:
        return [self.name, *self.aliases]


class EntityLinker:
    """Exact alias linking followed by conservative fuzzy matching."""

    def __init__(self, entries: Iterable[LinkEntry] = (), fuzzy_threshold: float = 0.85):
        self.fuzzy_threshold = float(fuzzy_threshold)
        self.entries: dict[str, LinkEntry] = {}
        self._surface_to_id: dict[str, str] = {}
        self._prefix: dict[str, list[str]] = {}
        for entry in entries:
            self.add(entry)

    def add(self, entry: LinkEntry) -> None:
        if not entry.canonical_id:
            entry.canonical_id = stable_entity_id(entry.name, entry.type)
        self.entries[entry.canonical_id] = entry
        for surface in entry.surfaces():
            key = canonical_key(surface)
            if not key:
                continue
            self._surface_to_id.setdefault(key, entry.canonical_id)
            self._prefix.setdefault(key[:1], []).append(key)

    def link(self, surface: str, entity_type: str = "") -> dict:
        # Preserve the submitted surface for audit; applying canonical_name here
        # would erase whether a match came from the explicit alias table.
        cleaned = " ".join(str(surface or "").split())
        key = canonical_key(cleaned)
        exact_id = self._surface_to_id.get(key)
        if exact_id:
            return self._result(cleaned, self.entries[exact_id], 1.0, "exact")

        alias_target = ALIASES.get(cleaned.lower())
        if alias_target:
            alias_id = self._surface_to_id.get(canonical_key(alias_target))
            if alias_id:
                return self._result(cleaned, self.entries[alias_id], 0.99, "alias")

        best_key, best_score = "", 0.0
        # The first-character bucket makes fuzzy matching practical for 50K+ KBs.
        candidates = self._prefix.get(key[:1], []) if key else []
        for candidate in candidates:
            candidate_id = self._surface_to_id[candidate]
            entry = self.entries[candidate_id]
            if entity_type and entry.type and entity_type != entry.type:
                continue
            length_ratio = min(len(key), len(candidate)) / max(len(key), len(candidate))
            if length_ratio < 0.65:
                continue
            score = SequenceMatcher(None, key, candidate).ratio()
            if score > best_score:
                best_key, best_score = candidate, score
        if best_key and best_score >= self.fuzzy_threshold:
            entry = self.entries[self._surface_to_id[best_key]]
            return self._result(cleaned, entry, round(best_score, 4), "fuzzy")

        return {
            "surface_form": cleaned,
            "canonical_id": stable_entity_id(cleaned, entity_type),
            "canonical_name": cleaned,
            "type": entity_type,
            "link_score": 0.0,
            "match_method": "unlinked_local_id",
            "kb_source": "MediGraph-local",
        }

    def link_entities(self, entities: Iterable[dict]) -> list[dict]:
        result = []
        for entity in entities:
            linked = self.link(str(entity.get("name", "")), str(entity.get("type", "")))
            result.append({**entity, **linked})
        return result

    @staticmethod
    def _result(surface: str, entry: LinkEntry, score: float, method: str) -> dict:
        return {
            "surface_form": surface,
            "canonical_id": entry.canonical_id,
            "canonical_name": entry.name,
            "type": entry.type,
            "link_score": score,
            "match_method": method,
            "kb_source": entry.source,
        }

    def to_dict(self) -> dict:
        return {
            "version": 1,
            "entries": [
                {
                    "canonical_id": entry.canonical_id,
                    "name": entry.name,
                    "type": entry.type,
                    "aliases": entry.aliases,
                    "source": entry.source,
                }
                for entry in sorted(self.entries.values(), key=lambda item: item.canonical_id)
            ],
        }

    def save(self, path: str | Path) -> str:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        return str(target)

    @classmethod
    def from_dict(cls, data: dict, fuzzy_threshold: float = 0.85) -> "EntityLinker":
        entries = [
            LinkEntry(
                canonical_id=str(item.get("canonical_id", "")),
                name=str(item.get("name", "")),
                type=str(item.get("type", "")),
                aliases=[str(alias) for alias in item.get("aliases", [])],
                source=str(item.get("source", "MediGraph")),
            )
            for item in data.get("entries", [])
            if item.get("name")
        ]
        return cls(entries, fuzzy_threshold=fuzzy_threshold)

    @classmethod
    def load(cls, path: str | Path, fuzzy_threshold: float = 0.85) -> "EntityLinker":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")), fuzzy_threshold)

    @classmethod
    def from_graph_json(cls, path: str | Path, source: str = "CM3KG") -> "EntityLinker":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        entries = []
        for node in data.get("nodes", []):
            name = str(node.get("canonical_name") or node.get("id") or "").strip()
            if not name:
                continue
            entity_type = str(node.get("type", ""))
            entries.append(
                LinkEntry(
                    canonical_id=str(node.get("canonical_id") or stable_entity_id(name, entity_type)),
                    name=name,
                    type=entity_type,
                    aliases=[str(value) for value in node.get("aliases", [])],
                    source=str(node.get("kb_source") or source),
                )
            )
        return cls(entries)
