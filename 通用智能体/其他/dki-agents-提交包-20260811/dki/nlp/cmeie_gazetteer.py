"""Leak-safe CMeIE training-gazetteer mention candidate generator.

The CMeEE tagger has nine biomedical entity types while CMeIE relation
arguments include values such as age groups, stages, locations and pathology
classes.  This module builds a typed mention lexicon from CMeIE *training*
rows only and finds literal occurrences with a trie.  It deliberately does
not learn from the rows it is asked to evaluate.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from dki.nlp.cmeie_pair import normalize_entity_type


_END = "\0"


class CmeieTrainGazetteer:
    """Typed surface-form matcher built from CMeIE training SPO arguments."""

    def __init__(self, rows: Iterable[dict[str, Any]], *, min_chars: int = 2):
        if min_chars < 1:
            raise ValueError("min_chars must be positive")
        typed_terms: dict[str, set[str]] = defaultdict(set)
        source_rows = 0
        for row in rows:
            source_rows += 1
            for spo in row.get("spo_list", []):
                for role in ("subject", "object"):
                    term = str(spo.get(role) or "").strip()
                    entity_type = normalize_entity_type(spo.get(f"{role}_type"))
                    if len(term) >= min_chars and entity_type:
                        typed_terms[term].add(entity_type)

        self.min_chars = min_chars
        self.source_rows = source_rows
        self.term_count = len(typed_terms)
        self.type_assignments = sum(len(types) for types in typed_terms.values())
        self._root: dict[str, Any] = {}
        for term, types in typed_terms.items():
            node = self._root
            for char in term:
                node = node.setdefault(char, {})
            node[_END] = tuple(sorted(types))

    def extract(self, text: str) -> list[dict[str, Any]]:
        """Return every exact typed training-lexicon hit, sorted stably."""
        found: set[tuple[int, int, str]] = set()
        for start in range(len(text)):
            node = self._root
            for end in range(start, len(text)):
                node = node.get(text[end])
                if node is None:
                    break
                types = node.get(_END)
                if types:
                    for entity_type in types:
                        found.add((start, end + 1, entity_type))
        return [
            {
                "name": text[start:end],
                "type": entity_type,
                "span": [start, end],
            }
            for start, end, entity_type in sorted(found)
        ]

    def provenance(self) -> dict[str, int | str]:
        return {
            "kind": "cmeie_train_gazetteer",
            "source_rows": self.source_rows,
            "min_chars": self.min_chars,
            "term_count": self.term_count,
            "type_assignments": self.type_assignments,
        }

    def augment_types(
        self, text: str, entities: Iterable[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Add train-lexicon CMeIE types to exact existing mention spans.

        Keeping the upstream spans avoids the candidate explosion caused by a
        free-running lexicon while still letting the relation model use CMeIE's
        richer argument-type signatures.
        """
        lexicon_types: dict[tuple[int, int, str], set[str]] = defaultdict(set)
        for hit in self.extract(text):
            span = hit["span"]
            lexicon_types[(span[0], span[1], hit["name"])].add(hit["type"])

        merged: set[tuple[int, int, str, str]] = set()
        for entity in entities:
            name = str(entity.get("name") or "")
            span = entity.get("span")
            if not name or not isinstance(span, (list, tuple)) or len(span) != 2:
                continue
            start, end = int(span[0]), int(span[1])
            types = {normalize_entity_type(entity.get("type"))}
            types.update(lexicon_types.get((start, end, name), set()))
            for entity_type in types:
                merged.add((start, end, name, entity_type))
        return [
            {"name": name, "type": entity_type, "span": [start, end]}
            for start, end, name, entity_type in sorted(merged)
        ]
