"""Neural small model for Task-1 orchestration policy prediction."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any
import warnings

from agents.task1_policy_model import (
    CAPABILITY_LABELS,
    KEYWORD_LABELS,
    OPERATOR_LABELS,
    WORKFLOW_LABELS,
    Task1PolicyPrediction,
    _feature_to_text,
    _labels_to_multihot,
    _load_policy_dataset_rows,
    _multihot_to_labels,
    _numeric_features,
    build_policy_features,
)


MODEL_VERSION = "task1_neural_orchestration_policy_v1_mlp"


@dataclass
class NeuralPolicyTrainConfig:
    hidden_layer_sizes: tuple[int, ...] = (192, 96)
    max_iter: int = 260
    random_state: int = 42


def _fit_mlp_classifier(x_train: Any, y_train: Any, config: NeuralPolicyTrainConfig) -> Any:
    from sklearn.neural_network import MLPClassifier

    class_counts = {}
    for item in y_train:
        class_counts[int(item)] = class_counts.get(int(item), 0) + 1
    early_stopping = len(class_counts) > 1 and min(class_counts.values()) >= 3
    clf = MLPClassifier(
        hidden_layer_sizes=config.hidden_layer_sizes,
        activation="relu",
        solver="adam",
        alpha=0.0005,
        batch_size=128,
        learning_rate_init=0.001,
        max_iter=config.max_iter,
        early_stopping=early_stopping,
        validation_fraction=0.12,
        n_iter_no_change=12,
        random_state=config.random_state,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Stochastic Optimizer: Maximum iterations.*")
        clf.fit(x_train, y_train)
    return clf


def _fit_multilabel_heads(x_train: Any, y_train: Any, config: NeuralPolicyTrainConfig) -> list[dict[str, Any]]:
    heads: list[dict[str, Any]] = []
    for column in range(y_train.shape[1]):
        values = y_train[:, column]
        unique = sorted({int(item) for item in values})
        if len(unique) == 1:
            heads.append({"constant": unique[0], "model": None})
            continue
        heads.append({"constant": None, "model": _fit_mlp_classifier(x_train, values, config)})
    return heads


def _predict_multilabel_heads(heads: list[dict[str, Any]], x: Any) -> Any:
    import numpy as np

    columns: list[Any] = []
    for head in heads:
        if head["model"] is None:
            columns.append(np.full(x.shape[0], int(head["constant"]), dtype=int))
        else:
            columns.append(head["model"].predict(x).astype(int))
    return np.vstack(columns).T


def _proba_confidence(clf: Any, x: Any) -> list[float]:
    if not hasattr(clf, "predict_proba"):
        return [0.5] * x.shape[0]
    probs = clf.predict_proba(x)
    return [float(max(row)) for row in probs]


def _estimate_mlp_parameters(clf: Any) -> int:
    if clf is None:
        return 0
    coefs = getattr(clf, "coefs_", []) or []
    intercepts = getattr(clf, "intercepts_", []) or []
    return int(sum(layer.size for layer in coefs) + sum(layer.size for layer in intercepts))


def _estimate_model_parameters(model: dict[str, Any]) -> int:
    total = 0
    for key, value in model.items():
        if key.endswith("_model"):
            total += _estimate_mlp_parameters(value)
        if key.endswith("_heads"):
            for head in value:
                total += _estimate_mlp_parameters(head.get("model"))
    return total


def _build_orchestration_json(row: dict[str, Any], prediction: Task1PolicyPrediction) -> dict[str, Any]:
    return {
        "workflow_type": prediction.workflow_type,
        "required_capabilities": prediction.required_capabilities,
        "operator_keywords": prediction.operator_keywords,
        "recommended_operator_ids": prediction.recommended_operator_ids,
        "quality_priority": prediction.quality_priority,
        "cost_tier": prediction.cost_tier,
        "requires_recovery": prediction.requires_recovery,
        "prefer_low_latency": prediction.prefer_low_latency,
        "strict_quality": prediction.strict_quality,
        "confidence": prediction.confidence,
        "source": prediction.source,
        "task": row.get("task"),
    }


def train_neural_policy_model(
    dataset_path: Path,
    model_path: Path,
    metrics_path: Path,
    *,
    feedback_dataset_paths: list[Path] | None = None,
    config: NeuralPolicyTrainConfig | None = None,
) -> dict[str, Any]:
    import joblib
    import numpy as np
    from scipy.sparse import hstack
    from sklearn.feature_extraction.text import HashingVectorizer
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MaxAbsScaler

    config = config or NeuralPolicyTrainConfig()
    rows, feedback_rows = _load_policy_dataset_rows(dataset_path, feedback_dataset_paths)
    if len(rows) < 100:
        raise ValueError("Neural Task-1 policy dataset must contain at least 100 rows")
    texts = [_feature_to_text(row["features"]) for row in rows]
    numeric = np.array([_numeric_features(row["features"]) for row in rows], dtype=float)
    workflow_y = np.array([WORKFLOW_LABELS.index(row["labels"]["workflow_type"]) for row in rows], dtype=int)
    capability_y = np.array([_labels_to_multihot(row["labels"]["required_capabilities"], CAPABILITY_LABELS) for row in rows], dtype=int)
    operator_y = np.array([_labels_to_multihot(row["labels"].get("recommended_operator_ids", []), OPERATOR_LABELS) for row in rows], dtype=int)
    keyword_y = np.array([_labels_to_multihot(row["labels"]["operator_keywords"], KEYWORD_LABELS) for row in rows], dtype=int)
    low_latency_y = np.array([1 if row["labels"].get("prefer_low_latency") else 0 for row in rows], dtype=int)
    strict_quality_y = np.array([1 if row["labels"].get("strict_quality") else 0 for row in rows], dtype=int)
    recovery_y = np.array([1 if row["labels"].get("requires_recovery") else 0 for row in rows], dtype=int)
    quality_priority_labels = ["standard", "high", "critical"]
    cost_tier_labels = ["low", "balanced", "quality_first"]
    quality_priority_y = np.array([quality_priority_labels.index(row["labels"].get("quality_priority", "standard")) for row in rows], dtype=int)
    cost_tier_y = np.array([cost_tier_labels.index(row["labels"].get("cost_tier", "balanced")) for row in rows], dtype=int)

    indices = np.arange(len(rows))
    train_idx, test_idx = train_test_split(indices, test_size=0.25, random_state=52, stratify=workflow_y)
    vectorizer = HashingVectorizer(n_features=3072, alternate_sign=False, ngram_range=(1, 3), norm="l2")
    scaler = MaxAbsScaler()
    text_x = vectorizer.transform(texts)
    numeric_x = scaler.fit_transform(numeric)
    x = hstack([text_x, numeric_x]).tocsr()

    workflow_model = _fit_mlp_classifier(x[train_idx], workflow_y[train_idx], config)
    low_latency_model = _fit_mlp_classifier(x[train_idx], low_latency_y[train_idx], config)
    strict_quality_model = _fit_mlp_classifier(x[train_idx], strict_quality_y[train_idx], config)
    recovery_model = _fit_mlp_classifier(x[train_idx], recovery_y[train_idx], config)
    quality_priority_model = _fit_mlp_classifier(x[train_idx], quality_priority_y[train_idx], config)
    cost_tier_model = _fit_mlp_classifier(x[train_idx], cost_tier_y[train_idx], config)
    capability_heads = _fit_multilabel_heads(x[train_idx], capability_y[train_idx], config)
    operator_heads = _fit_multilabel_heads(x[train_idx], operator_y[train_idx], config)
    keyword_heads = _fit_multilabel_heads(x[train_idx], keyword_y[train_idx], config)

    model = {
        "model_version": MODEL_VERSION,
        "vectorizer": vectorizer,
        "scaler": scaler,
        "workflow_model": workflow_model,
        "capability_heads": capability_heads,
        "operator_heads": operator_heads,
        "keyword_heads": keyword_heads,
        "low_latency_model": low_latency_model,
        "strict_quality_model": strict_quality_model,
        "recovery_model": recovery_model,
        "quality_priority_model": quality_priority_model,
        "cost_tier_model": cost_tier_model,
        "labels": {
            "workflow": WORKFLOW_LABELS,
            "capabilities": CAPABILITY_LABELS,
            "operators": OPERATOR_LABELS,
            "keywords": KEYWORD_LABELS,
            "quality_priority": quality_priority_labels,
            "cost_tier": cost_tier_labels,
        },
    }

    test_x = x[test_idx]
    workflow_pred = workflow_model.predict(test_x)
    capability_pred = _predict_multilabel_heads(capability_heads, test_x)
    operator_pred = _predict_multilabel_heads(operator_heads, test_x)
    keyword_pred = _predict_multilabel_heads(keyword_heads, test_x)
    low_latency_pred = low_latency_model.predict(test_x)
    strict_quality_pred = strict_quality_model.predict(test_x)
    recovery_pred = recovery_model.predict(test_x)
    quality_priority_pred = quality_priority_model.predict(test_x)
    cost_tier_pred = cost_tier_model.predict(test_x)

    ood_terms = ("不允许", "高吞吐", "OOD", "schema漂移", "无标准text", "隐私噪声", "长篇病历", "旧失败反馈")
    ood_idx = np.array([
        index for index, row in enumerate(rows)
        if row.get("source") == "datamate_runtime_feedback" or any(term in str(row.get("task", "")) for term in ood_terms)
    ], dtype=int)
    if len(ood_idx) > 0:
        train_set = {int(item) for item in train_idx}
        ood_idx = np.array([int(item) for item in ood_idx if int(item) not in train_set], dtype=int)

    def metric_block(eval_idx: Any) -> dict[str, float]:
        eval_x = x[eval_idx]
        return {
            "workflow_accuracy": round(float(accuracy_score(workflow_y[eval_idx], workflow_model.predict(eval_x))), 4),
            "capability_micro_f1": round(float(f1_score(capability_y[eval_idx], _predict_multilabel_heads(capability_heads, eval_x), average="micro", zero_division=0)), 4),
            "operator_micro_f1": round(float(f1_score(operator_y[eval_idx], _predict_multilabel_heads(operator_heads, eval_x), average="micro", zero_division=0)), 4),
            "keyword_micro_f1": round(float(f1_score(keyword_y[eval_idx], _predict_multilabel_heads(keyword_heads, eval_x), average="micro", zero_division=0)), 4),
            "recovery_accuracy": round(float(accuracy_score(recovery_y[eval_idx], recovery_model.predict(eval_x))), 4),
        }

    metrics = {
        "model_version": MODEL_VERSION,
        "dataset_path": str(dataset_path),
        "model_path": str(model_path),
        "sample_count": len(rows),
        "distillation_sample_count": len(rows) - len(feedback_rows),
        "runtime_feedback_sample_count": len(feedback_rows),
        "train_count": len(train_idx),
        "test_count": len(test_idx),
        "hidden_layer_sizes": list(config.hidden_layer_sizes),
        "max_iter": config.max_iter,
        "parameter_count_estimate": 0,
        "less_than_1b_parameters": True,
        **metric_block(test_idx),
        "low_latency_accuracy": round(float(accuracy_score(low_latency_y[test_idx], low_latency_pred)), 4),
        "strict_quality_accuracy": round(float(accuracy_score(strict_quality_y[test_idx], strict_quality_pred)), 4),
        "quality_priority_accuracy": round(float(accuracy_score(quality_priority_y[test_idx], quality_priority_pred)), 4),
        "cost_tier_accuracy": round(float(accuracy_score(cost_tier_y[test_idx], cost_tier_pred)), 4),
    }
    if len(ood_idx) >= 20:
        metrics.update({f"ood_{key}": value for key, value in metric_block(ood_idx).items()})
        metrics["ood_sample_count"] = int(len(ood_idx))
    metrics["parameter_count_estimate"] = _estimate_model_parameters(model)
    metrics["less_than_1b_parameters"] = metrics["parameter_count_estimate"] < 1_000_000_000
    model["metrics"] = metrics

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def load_neural_policy_model(model_path: Path) -> Any:
    import joblib

    if not model_path.exists():
        return None
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        return joblib.load(model_path)


def predict_neural_policy(model: Any, task: str, input_profile: dict[str, Any]) -> Task1PolicyPrediction | None:
    if not model:
        return None
    from scipy.sparse import hstack
    import numpy as np

    features = build_policy_features(task, input_profile)
    text_x = model["vectorizer"].transform([_feature_to_text(features)])
    numeric_x = model["scaler"].transform(np.array([_numeric_features(features)], dtype=float))
    x = hstack([text_x, numeric_x]).tocsr()
    workflow_index = int(model["workflow_model"].predict(x)[0])
    quality_priority_labels = model.get("labels", {}).get("quality_priority", ["standard", "high", "critical"])
    cost_tier_labels = model.get("labels", {}).get("cost_tier", ["low", "balanced", "quality_first"])
    confidence_parts = [
        *_proba_confidence(model["workflow_model"], x),
        *_proba_confidence(model["low_latency_model"], x),
        *_proba_confidence(model["strict_quality_model"], x),
        *_proba_confidence(model["recovery_model"], x),
        *_proba_confidence(model["quality_priority_model"], x),
        *_proba_confidence(model["cost_tier_model"], x),
    ]
    return Task1PolicyPrediction(
        workflow_type=WORKFLOW_LABELS[workflow_index],
        required_capabilities=_multihot_to_labels(_predict_multilabel_heads(model["capability_heads"], x)[0].tolist(), CAPABILITY_LABELS),
        operator_keywords=_multihot_to_labels(_predict_multilabel_heads(model["keyword_heads"], x)[0].tolist(), KEYWORD_LABELS),
        recommended_operator_ids=_multihot_to_labels(_predict_multilabel_heads(model["operator_heads"], x)[0].tolist(), OPERATOR_LABELS),
        prefer_low_latency=bool(int(model["low_latency_model"].predict(x)[0])),
        strict_quality=bool(int(model["strict_quality_model"].predict(x)[0])),
        requires_recovery=bool(int(model["recovery_model"].predict(x)[0])),
        quality_priority=str(quality_priority_labels[int(model["quality_priority_model"].predict(x)[0])]),
        cost_tier=str(cost_tier_labels[int(model["cost_tier_model"].predict(x)[0])]),
        avoid_operator_ids=[],
        failure_memory_matches=[],
        confidence=round(sum(confidence_parts) / len(confidence_parts), 4),
        source="trained_task1_neural_orchestration_policy_model",
    )


def predict_neural_orchestration_json(model: Any, row: dict[str, Any]) -> dict[str, Any] | None:
    prediction = predict_neural_policy(model, str(row.get("task") or ""), _features_to_input_profile(row.get("features", {})))
    return _build_orchestration_json(row, prediction) if prediction else None


def _features_to_input_profile(features: dict[str, Any]) -> dict[str, Any]:
    return {
        "format": features.get("format"),
        "structured": bool(features.get("structured")),
        "unstructured": bool(features.get("unstructured")),
        "batch": bool(features.get("batch")),
        "complexity": features.get("complexity"),
        "size_bytes": int(features.get("size_bytes") or 0),
        "record_count_estimate": int(features.get("record_count_estimate") or 0),
        "sampled_record_count": int(features.get("sampled_record_count") or 0),
        "schema_profile": {
            "field_count": int(features.get("schema_field_count") or 0),
            "schema_confidence": {
                "text_coverage": float(features.get("schema_text_coverage") or 0),
                "has_id": bool(features.get("schema_has_id")),
                "has_text": bool(features.get("schema_has_text")),
            },
        },
        "document_profile": {"document_count": int(features.get("document_count") or 0)},
        "quality_report": {"warning_count": int(features.get("quality_warning_count") or 0)},
        "batch_profile": {
            "file_count": int(features.get("batch_file_count") or 1),
            "structured_file_count": int(features.get("structured_file_count") or 0),
            "unstructured_file_count": int(features.get("unstructured_file_count") or 0),
        },
    }
