"""Tensorized KG candidate operators with CPU/NPU backends."""

from __future__ import annotations

import hashlib
import math
from typing import Any


VECTOR_DIM = 512


def tokens(text: str) -> list[str]:
    chars = [ch for ch in text.strip() if not ch.isspace()]
    output = chars[:]
    for size in (2, 3, 4):
        output.extend("".join(chars[index : index + size]) for index in range(max(0, len(chars) - size + 1)))
    return output


def hashed_vector(text: str, dim: int = VECTOR_DIM) -> list[float]:
    vector = [0.0] * dim
    for token in tokens(text):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[bucket] += sign * (1.0 + min(len(token), 4) * 0.15)
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else vector


def select_tensor_device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    import torch

    if requested in {"npu", "auto"}:
        try:
            import torch_npu  # noqa: F401
        except ModuleNotFoundError:
            if requested == "npu":
                raise RuntimeError("NPU requested but torch_npu is not installed")
            return "cpu"

        if torch.npu.is_available():
            return "npu"
    if requested == "npu":
        raise RuntimeError("NPU requested but torch.npu.is_available() is false")
    return "cpu"


def make_tensor(rows: list[list[float]], device: str):
    import torch

    tensor = torch.tensor(rows, dtype=torch.float32)
    return tensor.to("npu:0") if device == "npu" else tensor


def sync_tensor_device(device: str) -> None:
    if device == "npu":
        import torch

        torch.npu.synchronize()


def normalize_tensor_rows(tensor):
    return tensor / tensor.norm(dim=1, keepdim=True).clamp_min(1e-12)


def score_candidates(document_vectors: list[list[float]], candidate_vectors: list[list[float]], *, device: str):
    import torch

    selected_device = select_tensor_device(device)
    doc_tensor = make_tensor(document_vectors, selected_device)
    candidate_tensor = make_tensor(candidate_vectors, selected_device)
    sync_tensor_device(selected_device)
    scores = torch.matmul(doc_tensor, candidate_tensor.T)
    sync_tensor_device(selected_device)
    return scores


def score_candidates_topk(
    document_vectors: list[list[float]],
    candidate_vectors: list[list[float]],
    *,
    device: str,
    top_k: int = 16,
    normalize: bool = False,
):
    import torch

    selected_device = select_tensor_device(device)
    doc_tensor = make_tensor(document_vectors, selected_device)
    candidate_tensor = make_tensor(candidate_vectors, selected_device)
    if normalize:
        doc_tensor = normalize_tensor_rows(doc_tensor)
        candidate_tensor = normalize_tensor_rows(candidate_tensor)
    sync_tensor_device(selected_device)
    scores = torch.matmul(doc_tensor, candidate_tensor.T)
    k = min(max(1, int(top_k)), int(scores.shape[1]))
    values, indices = torch.topk(scores, k=k, dim=1)
    sync_tensor_device(selected_device)
    return values, indices


def score_tensor_topk(doc_tensor, candidate_tensor, *, top_k: int = 16, normalize: bool = False):
    import torch

    if normalize:
        doc_tensor = normalize_tensor_rows(doc_tensor)
        candidate_tensor = normalize_tensor_rows(candidate_tensor)
    scores = torch.matmul(doc_tensor, candidate_tensor.T)
    k = min(max(1, int(top_k)), int(scores.shape[1]))
    return torch.topk(scores, k=k, dim=1)


def rank_entity_candidates(
    document_vectors: list[list[float]],
    entity_candidates: list[tuple[str, str]],
    *,
    vector_dim: int = VECTOR_DIM,
    device: str = "auto",
):
    candidate_vectors = [hashed_vector(name, dim=vector_dim) for name, _ in entity_candidates]
    return score_candidates(document_vectors, candidate_vectors, device=device)


def score_relation_intents(
    document_vectors: list[list[float]],
    relation_catalog: dict[str, tuple[str, ...]],
    *,
    vector_dim: int = VECTOR_DIM,
    device: str = "auto",
):
    candidate_vectors = [hashed_vector(" ".join(triggers), dim=vector_dim) for triggers in relation_catalog.values()]
    return score_candidates(document_vectors, candidate_vectors, device=device)


def build_operator_descriptor() -> dict[str, Any]:
    return {
        "name": "tensorized_kg_candidate_ranker",
        "backend": "cpu_or_ascend_npu",
        "operators": ["entity_extract_candidate_rank", "relation_intent_score", "triple_generation_support"],
        "input": "medical document text/title batch",
        "output": "entity/relation score tensors used by KG extraction",
    }
