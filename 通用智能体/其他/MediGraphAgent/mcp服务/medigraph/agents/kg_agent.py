"""KGGen Agent (Task 2): raw documents -> knowledge graph.

Orchestrates the operator pipeline automatically for every document:
  chunker -> medical_ner -> medical_re -> triple_validator -> write to GraphStore.

Includes lightweight ReAct-style quality self-checks: if a chunk yields no
entities it is skipped; relation extraction is fed the chunk's entities so RE
stays grounded. Supports incremental building (call build() repeatedly).
"""
from __future__ import annotations

import hashlib
from typing import Any

from tqdm import tqdm

from medigraph.graph.base import GraphStore, get_graph_store
from medigraph.graph.vector_store import LocalVectorStore
from medigraph.operators.base import get_operator, load_default_operators


class KGGenAgent:
    def __init__(self, llm: Any | None = None, store: GraphStore | None = None,
                 build_vectors: bool = True):
        if llm is None:
            from medigraph.llm.client import LLMClient
            llm = LLMClient()
        self.llm = llm
        load_default_operators(llm=llm)
        self.store = store or get_graph_store()
        self.chunker = get_operator("chunker")
        self.ner = get_operator("medical_ner")
        self.re = get_operator("medical_re")
        self.validator = get_operator("triple_validator")
        # Vector index over chunks for hybrid GraphRAG (graph + vector retrieval).
        self.build_vectors = build_vectors
        self.vector_store = LocalVectorStore() if build_vectors else None

    def build(self, documents: list[dict], verbose: bool = True, max_chunks_per_doc: int | None = None) -> dict:
        """documents: [{fileName, text, ...}]. Returns build stats.

        max_chunks_per_doc caps how many chunks of each document are processed
        (useful to bound runtime / API cost on slow models).
        """
        n_chunks = n_entities = n_candidate = n_valid = n_rejected = 0
        route_counts: dict[str, int] = {}
        if hasattr(self.store, "begin_revision"):
            self.store.begin_revision(f"kg-build:{len(documents)}-documents")

        for doc in documents:
            source = doc.get("fileName", "unknown")
            text = doc.get("text", "") or ""
            if not text.strip():
                continue
            chunks = self.chunker.run({"text": text}).get("chunks", [])
            if max_chunks_per_doc:
                chunks = chunks[:max_chunks_per_doc]
            # Index chunks for vector retrieval (best-effort; degrades if no API).
            if self.vector_store is not None and chunks:
                vecs = self.llm.embed(chunks)
                self.vector_store.add(chunks, [source] * len(chunks), vecs)
            iterator = tqdm(chunks, desc=f"KG:{source[:24]}", disable=not verbose, leave=False)
            source_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            for chunk_index, chunk in enumerate(iterator):
                n_chunks += 1
                ner_result = self.ner.run({"text": chunk})
                ents = ner_result.get("entities", [])
                ner_route = ner_result.get("routing", {}).get("level", "unknown")
                route_counts[f"ner:{ner_route}"] = route_counts.get(f"ner:{ner_route}", 0) + 1
                if not ents:  # self-check: nothing to relate
                    continue
                n_entities += len(ents)
                for e in ents:
                    self.store.upsert_entity(
                        e["name"],
                        e["type"],
                        confidence=e.get("confidence", 1.0),
                        canonical_id=e.get("canonical_id", ""),
                        canonical_name=e.get("canonical_name", e["name"]),
                        link_score=e.get("link_score", 0.0),
                        kb_source=e.get("kb_source", ""),
                    )

                re_result = self.re.run({"text": chunk, "entities": ents})
                triples = re_result.get("triples", [])
                re_route = re_result.get("routing", {}).get("level", "unknown")
                route_counts[f"re:{re_route}"] = route_counts.get(f"re:{re_route}", 0) + 1
                n_candidate += len(triples)
                checked = self.validator.run({"triples": triples, "text": chunk})
                valid = checked.get("valid", [])
                n_valid += len(valid)
                n_rejected += len(checked.get("rejected", []))
                chunk_id = hashlib.sha256(
                    f"{source}:{chunk_index}:{chunk}".encode("utf-8")
                ).hexdigest()[:20]
                for t in valid:
                    self.store.upsert_triple(
                        head=t["head"], head_type=t["head_type"], relation=t["relation"],
                        tail=t["tail"], tail_type=t["tail_type"],
                        confidence=t.get("confidence", 1.0), source=source,
                        operator_version=f"{t.get('extractor', 'medical_re')}/{t.get('model_version', '2.0.0')}",
                        source_doc=source,
                        source_hash=source_hash,
                        chunk_id=chunk_id,
                        extractor=t.get("extractor", "medical_re"),
                        model_version=t.get("model_version", "2.0.0"),
                        evidence=t.get("evidence", chunk[:240]),
                        cmeie_relation=t.get("cmeie_relation", ""),
                        cmeie_predicate=t.get("cmeie_predicate", ""),
                    )

        delta = self.store.commit_revision() if hasattr(self.store, "commit_revision") else {}
        build_stats = {
            "documents": len(documents),
            "chunks": n_chunks,
            "entities_extracted": n_entities,
            "candidate_triples": n_candidate,
            "valid_triples": n_valid,
            "rejected_triples": n_rejected,
            "graph": self.store.stats(),
            "graph_audit": self.store.audit() if hasattr(self.store, "audit") else {},
            "incremental_delta": delta,
            "routing": route_counts,
            "vector_chunks_indexed": self.vector_store.size if self.vector_store else 0,
            "llm": self.llm.stats.summary(),
        }
        return build_stats
