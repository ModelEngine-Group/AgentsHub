from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_MODEL_PATH = Path("artifacts/task1/models/flow_planner/model.json")


def tokenize(text: str) -> list[str]:
    lowered = text.lower()
    ascii_tokens = re.findall(r"[a-z0-9_]+", lowered)
    cjk_tokens = re.findall(r"[\u4e00-\u9fff]{1,4}", lowered)
    return ascii_tokens + cjk_tokens


def sample_text(sample: dict[str, Any]) -> str:
    profile = sample.get("profile", {})
    parts = [
        str(sample.get("task", "")),
        str(sample.get("data_mode", "")),
        str(sample.get("optimization_goal", "")),
        " ".join(str(item) for item in profile.get("missing_fields", [])),
        " ".join(str(item) for item in profile.get("noise_types", [])),
        f"records_{profile.get('record_bucket', '')}",
        f"text_coverage_{profile.get('text_coverage_bucket', '')}",
        f"structured_coverage_{profile.get('structured_coverage_bucket', '')}",
    ]
    return " ".join(parts)


class FlowPlannerModel:
    """Tiny multilabel Naive Bayes model for data-flow planning.

    It predicts which processing operators should be selected from a task and
    a lightweight data profile. The model is intentionally dependency-free so
    the contest demo remains reproducible on a clean Python installation.
    """

    def __init__(self) -> None:
        self.labels: list[str] = []
        self.priors: dict[str, float] = {}
        self.positive_token_counts: dict[str, dict[str, int]] = {}
        self.negative_token_counts: dict[str, dict[str, int]] = {}
        self.positive_totals: dict[str, int] = {}
        self.negative_totals: dict[str, int] = {}
        self.vocabulary: list[str] = []
        self.thresholds: dict[str, float] = {}

    def fit(self, samples: list[dict[str, Any]]) -> None:
        labels = sorted({label for sample in samples for label in sample["labels"]})
        vocab: set[str] = set()
        positive_counts = {label: Counter() for label in labels}
        negative_counts = {label: Counter() for label in labels}
        positive_docs = Counter()

        for sample in samples:
            tokens = tokenize(sample_text(sample))
            vocab.update(tokens)
            sample_labels = set(sample["labels"])
            for label in labels:
                if label in sample_labels:
                    positive_counts[label].update(tokens)
                    positive_docs[label] += 1
                else:
                    negative_counts[label].update(tokens)

        total_docs = len(samples)
        self.labels = labels
        self.vocabulary = sorted(vocab)
        self.priors = {
            label: (positive_docs[label] + 1) / (total_docs + 2)
            for label in labels
        }
        self.positive_token_counts = {label: dict(counts) for label, counts in positive_counts.items()}
        self.negative_token_counts = {label: dict(counts) for label, counts in negative_counts.items()}
        self.positive_totals = {label: sum(counts.values()) for label, counts in positive_counts.items()}
        self.negative_totals = {label: sum(counts.values()) for label, counts in negative_counts.items()}
        self.thresholds = {label: 0.5 for label in labels}

    def predict_scores(self, sample: dict[str, Any]) -> dict[str, float]:
        tokens = tokenize(sample_text(sample))
        vocab_size = max(len(self.vocabulary), 1)
        scores: dict[str, float] = {}
        for label in self.labels:
            prior = min(max(self.priors[label], 1e-6), 1 - 1e-6)
            log_pos = math.log(prior)
            log_neg = math.log(1 - prior)
            pos_counts = self.positive_token_counts[label]
            neg_counts = self.negative_token_counts[label]
            pos_total = self.positive_totals[label]
            neg_total = self.negative_totals[label]
            for token in tokens:
                log_pos += math.log((pos_counts.get(token, 0) + 1) / (pos_total + vocab_size))
                log_neg += math.log((neg_counts.get(token, 0) + 1) / (neg_total + vocab_size))
            probability = 1 / (1 + math.exp(min(max(log_neg - log_pos, -60), 60)))
            scores[label] = round(probability, 4)
        return scores

    def predict(self, sample: dict[str, Any]) -> list[str]:
        scores = self.predict_scores(sample)
        selected = [label for label in self.labels if scores[label] >= self.thresholds.get(label, 0.5)]
        if not selected:
            selected = ["raw_record_profiler", "medical_text_loader"]
        return self._order_tools(selected)

    def _order_tools(self, labels: list[str]) -> list[str]:
        canonical_order = [
            "raw_record_profiler",
            "medical_record_normalizer",
            "medical_text_loader",
            "medical_text_cleaner",
            "medical_text_segmenter",
            "table_schema_mapper",
            "deduplication_operator",
            "quality_anomaly_detector",
            "batch_scheduler",
        ]
        selected = set(labels)
        return [tool for tool in canonical_order if tool in selected]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": "multilabel_naive_bayes",
            "parameter_count_estimate": self.parameter_count_estimate(),
            "labels": self.labels,
            "priors": self.priors,
            "positive_token_counts": self.positive_token_counts,
            "negative_token_counts": self.negative_token_counts,
            "positive_totals": self.positive_totals,
            "negative_totals": self.negative_totals,
            "vocabulary": self.vocabulary,
            "thresholds": self.thresholds,
        }

    def parameter_count_estimate(self) -> int:
        return len(self.labels) * max(len(self.vocabulary), 1) * 2 + len(self.labels)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FlowPlannerModel":
        model = cls()
        model.labels = list(payload["labels"])
        model.priors = {str(k): float(v) for k, v in payload["priors"].items()}
        model.positive_token_counts = {
            str(label): {str(token): int(count) for token, count in counts.items()}
            for label, counts in payload["positive_token_counts"].items()
        }
        model.negative_token_counts = {
            str(label): {str(token): int(count) for token, count in counts.items()}
            for label, counts in payload["negative_token_counts"].items()
        }
        model.positive_totals = {str(k): int(v) for k, v in payload["positive_totals"].items()}
        model.negative_totals = {str(k): int(v) for k, v in payload["negative_totals"].items()}
        model.vocabulary = list(payload["vocabulary"])
        model.thresholds = {str(k): float(v) for k, v in payload.get("thresholds", {}).items()}
        return model

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path = DEFAULT_MODEL_PATH) -> "FlowPlannerModel":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(payload)


def evaluate(model: FlowPlannerModel, samples: list[dict[str, Any]]) -> dict[str, Any]:
    label_tp = defaultdict(int)
    label_fp = defaultdict(int)
    label_fn = defaultdict(int)
    exact = 0
    for sample in samples:
        gold = set(sample["labels"])
        pred = set(model.predict(sample))
        exact += int(gold == pred)
        for label in model.labels:
            if label in gold and label in pred:
                label_tp[label] += 1
            elif label not in gold and label in pred:
                label_fp[label] += 1
            elif label in gold and label not in pred:
                label_fn[label] += 1

    tp = sum(label_tp.values())
    fp = sum(label_fp.values())
    fn = sum(label_fn.values())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "sample_count": len(samples),
        "exact_match": round(exact / len(samples), 4) if samples else 0.0,
        "micro_precision": round(precision, 4),
        "micro_recall": round(recall, 4),
        "micro_f1": round(f1, 4),
        "parameter_count_estimate": model.parameter_count_estimate(),
    }
