"""Neural joint entity-relation extractor (GPLinker / GlobalPointer).

This is the trained L1 *neural* fast path that supersedes the deterministic
lexicon baseline for accuracy-critical extraction.  A single model jointly
predicts:

  * typed entity spans (GlobalPointer with one head per entity type), and
  * (subject, predicate, object) triples (head-to-head and tail-to-tail
    GlobalPointers over the full CMeIE-V2 predicate space),

so there is **no NER -> RE error-propagation pipeline**: entities and relations
are decoded from the same forward pass.

The module is dependency-light at import time: ``torch``/``transformers`` are
imported lazily inside the classes that need them, so the rest of the package
(and the CPU lexicon path) keeps working when the ML stack is absent.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# ----------------------------------------------------------------------------- #
# Model definition (torch).  Imported lazily by training / inference entrypoints.
# ----------------------------------------------------------------------------- #
def _build_torch_modules():
    import torch
    from torch import nn

    class GlobalPointer(nn.Module):
        """Span-scoring head (Su Jianlin, 2021) with optional rotary embeddings."""

        def __init__(self, hidden_size: int, heads: int, head_size: int = 64,
                     rope: bool = True, tril_mask: bool = True):
            super().__init__()
            self.heads = heads
            self.head_size = head_size
            self.rope = rope
            self.tril_mask = tril_mask
            self.dense = nn.Linear(hidden_size, heads * head_size * 2)

        @staticmethod
        def _rope(pos_dim: int, seq_len: int, device, dtype):
            idx = torch.arange(0, pos_dim, 2, dtype=torch.float32, device=device)
            theta = 1.0 / (10000 ** (idx / pos_dim))
            pos = torch.arange(seq_len, dtype=torch.float32, device=device)
            freqs = torch.einsum("n,d->nd", pos, theta)  # [L, pos_dim/2]
            emb = torch.stack([freqs, freqs], dim=-1).reshape(seq_len, pos_dim)
            return emb.sin().to(dtype), emb.cos().to(dtype)

        def forward(self, hidden: "torch.Tensor", mask: "torch.Tensor"):
            b, l, _ = hidden.shape
            x = self.dense(hidden).view(b, l, self.heads, 2 * self.head_size)
            qw, kw = x[..., : self.head_size], x[..., self.head_size:]
            if self.rope:
                sin, cos = self._rope(self.head_size, l, hidden.device, hidden.dtype)
                sin = sin[None, :, None, :]
                cos = cos[None, :, None, :]

                def rotate(t):
                    t2 = torch.stack([-t[..., 1::2], t[..., 0::2]], dim=-1).reshape_as(t)
                    return t * cos + t2 * sin

                qw, kw = rotate(qw), rotate(kw)
            logits = torch.einsum("blhd,bmhd->bhlm", qw, kw)  # [B, heads, L, L]
            pad = mask[:, None, None, :] * mask[:, None, :, None]
            logits = logits - (1 - pad) * 1e12
            if self.tril_mask:
                tril = torch.tril(torch.ones(l, l, device=hidden.device), -1).bool()
                logits = logits - tril[None, None] * 1e12
            return logits / (self.head_size ** 0.5)

    class GPLinker(nn.Module):
        def __init__(self, encoder_name: str, num_entity_types: int,
                     num_predicates: int, head_size: int = 64):
            super().__init__()
            from transformers import AutoModel

            self.encoder = AutoModel.from_pretrained(encoder_name)
            hidden = self.encoder.config.hidden_size
            self.entity = GlobalPointer(hidden, num_entity_types, head_size,
                                        rope=True, tril_mask=True)
            self.head = GlobalPointer(hidden, num_predicates, head_size,
                                      rope=False, tril_mask=False)
            self.tail = GlobalPointer(hidden, num_predicates, head_size,
                                      rope=False, tril_mask=False)

        def forward(self, input_ids, attention_mask, token_type_ids=None):
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            h = out.last_hidden_state
            mask = attention_mask.float()
            return (
                self.entity(h, mask),
                self.head(h, mask),
                self.tail(h, mask),
            )

    def multilabel_cce(y_pred: "torch.Tensor", y_true: "torch.Tensor") -> "torch.Tensor":
        """Sparse-free multilabel categorical cross-entropy over the last dim."""
        y_pred = (1 - 2 * y_true) * y_pred
        y_pred_neg = y_pred - y_true * 1e12
        y_pred_pos = y_pred - (1 - y_true) * 1e12
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
        neg = torch.logsumexp(y_pred_neg, dim=-1)
        pos = torch.logsumexp(y_pred_pos, dim=-1)
        return (neg + pos).mean()

    return GlobalPointer, GPLinker, multilabel_cce


# ----------------------------------------------------------------------------- #
# Inference wrapper used by the extraction cascade.
# ----------------------------------------------------------------------------- #
class NeuralGPLinkerExtractor:
    """Loads a trained checkpoint and emits entities/triples for the cascade."""

    def __init__(self, model_dir: str | Path, device: str | None = None,
                 max_length: int = 256):
        import os

        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        import torch
        from transformers import AutoTokenizer
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        if hasattr(hf_logging, "disable_progress_bar"):
            hf_logging.disable_progress_bar()

        self.dir = Path(model_dir)
        cfg = json.loads((self.dir / "extractor_config.json").read_text(encoding="utf-8"))
        self.entity_types: list[str] = cfg["entity_types"]
        self.predicates: list[str] = cfg["predicates"]
        self.pred_schema: dict[str, dict] = cfg.get("predicate_schema", {})
        self.encoder_name: str = cfg["encoder_name"]
        self.max_length = int(cfg.get("max_length", max_length))
        self.max_span = int(cfg.get("max_span", 30))
        self.version = cfg.get("version", "neural-gplinker-1.0.0")
        encoder_name = Path(self.encoder_name)
        if not encoder_name.is_absolute() and not encoder_name.exists():
            try:
                from config.settings import PROJECT_ROOT

                candidate = PROJECT_ROOT / encoder_name
                if candidate.exists():
                    self.encoder_name = str(candidate)
            except Exception:
                pass

        _, GPLinker, _ = _build_torch_modules()
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.dir), use_fast=True)
        self.model = GPLinker(self.encoder_name, len(self.entity_types),
                              len(self.predicates))
        state = torch.load(self.dir / "pytorch_model.bin", map_location="cpu")
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()

    @classmethod
    def available(cls, model_dir: str | Path) -> bool:
        d = Path(model_dir)
        return (d / "extractor_config.json").exists() and (d / "pytorch_model.bin").exists()

    def _encode(self, text: str):
        enc = self.tokenizer(text[: self.max_length * 4], return_offsets_mapping=True,
                             max_length=self.max_length, truncation=True,
                             return_tensors="pt")
        offsets = enc.pop("offset_mapping")[0].tolist()
        enc = {k: v.to(self.device) for k, v in enc.items()}
        return enc, offsets

    def extract(self, text: str, threshold: float = 0.0,
                rel_threshold: float | None = None) -> dict[str, list[dict]]:
        torch = self.torch
        text = str(text or "")
        if not text.strip():
            return {"entities": [], "triples": []}
        enc, offsets = self._encode(text)
        with torch.no_grad():
            ent_logit, head_logit, tail_logit = self.model(
                enc["input_ids"], enc["attention_mask"],
                enc.get("token_type_ids"))
        ent_logit = ent_logit[0]
        head_logit = head_logit[0]
        tail_logit = tail_logit[0]

        def span_text(ts: int, te: int) -> tuple[str, int, int]:
            if ts >= len(offsets) or te >= len(offsets):
                return "", -1, -1
            cs = offsets[ts][0]
            ce = offsets[te][1]
            return text[cs:ce], cs, ce

        # --- typed entities --------------------------------------------------
        entities: list[dict] = []
        seen: set[tuple[int, int, str]] = set()
        for ti in range(len(self.entity_types)):
            idx = (ent_logit[ti] > threshold).nonzero(as_tuple=False).tolist()
            for ts, te in idx:
                surface, cs, ce = span_text(ts, te)
                if not surface.strip() or (cs, ce, self.entity_types[ti]) in seen:
                    continue
                seen.add((cs, ce, self.entity_types[ti]))
                entities.append({
                    "name": surface, "type": self.entity_types[ti],
                    "start": cs, "end": ce,
                    "confidence": round(float(torch.sigmoid(ent_logit[ti, ts, te])), 3),
                    "extractor": "neural_gplinker", "model_version": self.version,
                    "canonical_id": "", "training_source": "CMeIE-V2",
                })

        # --- triples (entity-gated GPLinker decode) --------------------------
        # Candidate argument spans are the typed entity-head predictions (keeps
        # precision high); a triple needs the subject/object spans linked by both
        # the head head (sh,oh) and the tail head (st,ot).  The relation heads use
        # a separate, typically lower threshold so triple *recall* can be raised
        # without admitting low-quality entity spans.
        if rel_threshold is None:
            rel_threshold = threshold
        ent_spans: list[tuple[int, int]] = []
        for ti in range(len(self.entity_types)):
            for ts, te in (ent_logit[ti] > threshold).nonzero(as_tuple=False).tolist():
                ent_spans.append((ts, te))
        ent_spans = sorted(set(ent_spans))
        starts: dict[int, list[int]] = {}
        ends: dict[int, list[int]] = {}
        for (s, e) in ent_spans:
            starts.setdefault(s, []).append(e)

        triples: list[dict] = []
        tkey: set[tuple[str, str, str]] = set()
        for p, predicate in enumerate(self.predicates):
            head_pairs = {(int(a), int(b)) for a, b in
                          (head_logit[p] > rel_threshold).nonzero(as_tuple=False).tolist()}
            if not head_pairs:
                continue
            tail_set = {(int(a), int(b)) for a, b in
                        (tail_logit[p] > rel_threshold).nonzero(as_tuple=False).tolist()}
            if not tail_set:
                continue
            for (sh, oh) in head_pairs:
                for st in starts.get(sh, ()):       # subject span (sh, st) is an entity
                    for ot in starts.get(oh, ()):   # object span (oh, ot) is an entity
                        if (st, ot) not in tail_set:
                            continue
                        subj, _, _ = span_text(sh, st)
                        obj, _, _ = span_text(oh, ot)
                        if not subj.strip() or not obj.strip():
                            continue
                        key = (subj, predicate, obj)
                        if key in tkey:
                            continue
                        tkey.add(key)
                        schema = self.pred_schema.get(predicate, {})
                        triples.append({
                            "head": subj, "head_type": schema.get("subject_type_en", ""),
                            "relation": f"cmeie:{schema.get('key', '')}",
                            "predicate": predicate,
                            "tail": obj, "tail_type": schema.get("object_type_en", ""),
                            "confidence": round(float(torch.sigmoid(head_logit[p, sh, oh])), 3),
                            "extractor": "neural_gplinker",
                            "model_version": self.version,
                        })
        return {"entities": entities, "triples": triples}
