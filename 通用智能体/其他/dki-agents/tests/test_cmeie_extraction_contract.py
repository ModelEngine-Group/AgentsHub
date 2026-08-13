from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from dki.nlp.cmeie_extraction_contract import (
    EXTRACTION_CONTRACT_FILENAME,
    EXTRACTION_CONTRACT_KIND,
    EXTRACTION_CONTRACT_SCHEMA_VERSION,
    ExtractionContractError,
    load_and_validate_extraction_contract,
    publish_extraction_contract_once,
)


VOCABULARY = {"[PAD]": 0, "药": 2, "病": 1}
ENTITY_LABELS = ["subject:疾病", "object:药物"]
PREDICATES = ["治疗", "预防"]
SCHEMA_CONSTRAINTS = {
    "治疗": [("subject:疾病", "object:药物")],
    "预防": [("subject:疾病", "object:药物")],
}


def _directory(tmp_path: Path) -> Path:
    checkpoint = tmp_path / "epoch-01"
    checkpoint.mkdir(parents=True)
    return checkpoint


def _publish(checkpoint: Path):
    return publish_extraction_contract_once(
        checkpoint,
        tokenizer_vocabulary=VOCABULARY,
        entity_labels=ENTITY_LABELS,
        predicates=PREDICATES,
        schema_constraints=SCHEMA_CONSTRAINTS,
    )


def _canonical(payload: dict) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _rewrite(checkpoint: Path, payload: dict) -> str:
    raw = _canonical(payload)
    (checkpoint / EXTRACTION_CONTRACT_FILENAME).write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def test_publish_and_bound_load_valid_contract(tmp_path: Path):
    checkpoint = _directory(tmp_path)
    binding = _publish(checkpoint)
    contract = checkpoint / EXTRACTION_CONTRACT_FILENAME
    raw = contract.read_bytes()
    payload = json.loads(raw)

    assert raw == _canonical(payload)
    assert payload == {
        "schema_version": EXTRACTION_CONTRACT_SCHEMA_VERSION,
        "kind": EXTRACTION_CONTRACT_KIND,
        "entity_labels": ENTITY_LABELS,
        "predicates": PREDICATES,
        "schema_constraints": {
            key: [list(pair) for pair in pairs]
            for key, pairs in SCHEMA_CONSTRAINTS.items()
        },
        "max_length": 320,
        "pointer_variant": "efficient",
        "head_reduction": "balanced",
        "threshold_values": [-0.5, 0.0, 0.5],
        "grid_size": 54,
    }
    assert set(binding) == {
        "extraction_contract",
        "extraction_contract_sha256",
        "tokenizer_vocabulary_sha256",
        "label_table_sha256",
        "schema_constraints_sha256",
        "semantic_contract_sha256",
    }
    assert binding["extraction_contract_sha256"] == hashlib.sha256(
        raw
    ).hexdigest()
    assert (
        load_and_validate_extraction_contract(
            checkpoint,
            tokenizer_vocabulary=dict(reversed(list(VOCABULARY.items()))),
            expected_binding=binding,
        )
        == binding
    )


def test_contract_must_precede_checkpoint_bundle_manifest(tmp_path: Path):
    checkpoint = _directory(tmp_path)
    (checkpoint / "checkpoint_bundle.json").write_text(
        "{}\n",
        encoding="utf-8",
    )
    with pytest.raises(
        ExtractionContractError,
        match="before the checkpoint bundle manifest",
    ):
        _publish(checkpoint)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("schema_version", True, "schema_version"),
        ("kind", "other", "kind"),
        ("max_length", True, "max_length"),
        ("max_length", 319, "max_length"),
        ("pointer_variant", "classic", "pointer_variant"),
        ("head_reduction", "mean", "head_reduction"),
        ("grid_size", 53, "grid_size"),
        (
            "threshold_values",
            [-0.5, float("nan"), 0.5],
            "threshold_values",
        ),
        (
            "threshold_values",
            [-0.5, False, 0.5],
            "threshold_values",
        ),
        (
            "threshold_values",
            [-0.5, 0, 0.5],
            "threshold_values",
        ),
    ],
)
def test_load_rejects_any_invalid_fixed_field(
    tmp_path: Path,
    field: str,
    replacement,
    message: str,
):
    checkpoint = _directory(tmp_path)
    binding = _publish(checkpoint)
    payload = json.loads(
        (checkpoint / EXTRACTION_CONTRACT_FILENAME).read_text(
            encoding="utf-8",
        )
    )
    payload[field] = replacement
    if field == "threshold_values" and replacement[1] != replacement[1]:
        raw = (
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        (checkpoint / EXTRACTION_CONTRACT_FILENAME).write_bytes(raw)
    else:
        _rewrite(checkpoint, payload)

    with pytest.raises(ExtractionContractError, match=message):
        load_and_validate_extraction_contract(
            checkpoint,
            tokenizer_vocabulary=VOCABULARY,
            expected_binding=binding,
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        (
            "entity_labels",
            ["subject:疾病", "subject:疾病"],
        ),
        ("entity_labels", ["subject:疾病", " "]),
        ("predicates", ["治疗", "治疗"]),
        ("predicates", ["治疗", 1]),
        (
            "schema_constraints",
            {"未知": [["subject:疾病", "object:药物"]]},
        ),
        (
            "schema_constraints",
            {
                "治疗": [
                    ["subject:疾病", "object:药物"],
                    ["subject:疾病", "object:药物"],
                ]
            },
        ),
        (
            "schema_constraints",
            {"治疗": [["subject:疾病", "object:未知"]]},
        ),
        ("schema_constraints", {"治疗": []}),
    ],
)
def test_contract_rejects_empty_wrong_type_or_duplicate_semantics(
    tmp_path: Path,
    field: str,
    replacement,
):
    checkpoint = _directory(tmp_path)
    binding = _publish(checkpoint)
    payload = json.loads(
        (checkpoint / EXTRACTION_CONTRACT_FILENAME).read_text(
            encoding="utf-8",
        )
    )
    payload[field] = replacement
    _rewrite(checkpoint, payload)

    with pytest.raises(ExtractionContractError):
        load_and_validate_extraction_contract(
            checkpoint,
            tokenizer_vocabulary=VOCABULARY,
            expected_binding=binding,
        )


def test_contract_rejects_extra_field_and_noncanonical_key_order(
    tmp_path: Path,
):
    extra_checkpoint = _directory(tmp_path / "extra")
    extra_binding = _publish(extra_checkpoint)
    extra_payload = json.loads(
        (
            extra_checkpoint / EXTRACTION_CONTRACT_FILENAME
        ).read_text(encoding="utf-8")
    )
    extra_payload["receipt_sha256"] = "0" * 64
    _rewrite(extra_checkpoint, extra_payload)
    with pytest.raises(ExtractionContractError, match="field set"):
        load_and_validate_extraction_contract(
            extra_checkpoint,
            tokenizer_vocabulary=VOCABULARY,
            expected_binding=extra_binding,
        )

    order_checkpoint = _directory(tmp_path / "order")
    order_binding = _publish(order_checkpoint)
    payload = json.loads(
        (order_checkpoint / EXTRACTION_CONTRACT_FILENAME).read_text(
            encoding="utf-8",
        )
    )
    reversed_payload = dict(reversed(list(payload.items())))
    noncanonical = (
        json.dumps(
            reversed_payload,
            ensure_ascii=False,
            sort_keys=False,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    assert noncanonical != _canonical(payload)
    (order_checkpoint / EXTRACTION_CONTRACT_FILENAME).write_bytes(
        noncanonical
    )
    with pytest.raises(ExtractionContractError, match="canonical JSON"):
        load_and_validate_extraction_contract(
            order_checkpoint,
            tokenizer_vocabulary=VOCABULARY,
            expected_binding=order_binding,
        )


def test_contract_rejects_semantic_list_reordering(tmp_path: Path):
    checkpoint = _directory(tmp_path)
    binding = _publish(checkpoint)
    payload = json.loads(
        (checkpoint / EXTRACTION_CONTRACT_FILENAME).read_text(
            encoding="utf-8",
        )
    )
    payload["predicates"].reverse()
    payload["schema_constraints"] = {
        predicate: payload["schema_constraints"][predicate]
        for predicate in payload["predicates"]
    }
    changed_file_sha = _rewrite(checkpoint, payload)
    changed_file_binding = {
        **binding,
        "extraction_contract_sha256": changed_file_sha,
    }

    with pytest.raises(
        ExtractionContractError,
        match="label_table_sha256, semantic_contract_sha256",
    ):
        load_and_validate_extraction_contract(
            checkpoint,
            tokenizer_vocabulary=VOCABULARY,
            expected_binding=changed_file_binding,
        )


def test_contract_rejects_duplicate_json_key(tmp_path: Path):
    checkpoint = _directory(tmp_path)
    binding = _publish(checkpoint)
    contract = checkpoint / EXTRACTION_CONTRACT_FILENAME
    text = contract.read_text(encoding="utf-8")
    text = text.replace(
        '"schema_version":1',
        '"schema_version":1,"schema_version":1',
        1,
    )
    contract.write_text(text, encoding="utf-8")

    with pytest.raises(
        ExtractionContractError,
        match="cannot load bound",
    ):
        load_and_validate_extraction_contract(
            checkpoint,
            tokenizer_vocabulary=VOCABULARY,
            expected_binding=binding,
        )


@pytest.mark.parametrize(
    "field",
    [
        "extraction_contract_sha256",
        "tokenizer_vocabulary_sha256",
        "label_table_sha256",
        "schema_constraints_sha256",
        "semantic_contract_sha256",
    ],
)
def test_load_rejects_each_receipt_hash_tampering(
    tmp_path: Path,
    field: str,
):
    checkpoint = _directory(tmp_path)
    binding = _publish(checkpoint)
    tampered = {**binding, field: "0" * 64}
    assert tampered[field] != binding[field]

    with pytest.raises(ExtractionContractError, match=field):
        load_and_validate_extraction_contract(
            checkpoint,
            tokenizer_vocabulary=VOCABULARY,
            expected_binding=tampered,
        )


def test_load_requires_every_external_hash_and_rejects_conflicts(
    tmp_path: Path,
):
    checkpoint = _directory(tmp_path)
    binding = _publish(checkpoint)
    incomplete = dict(binding)
    del incomplete["semantic_contract_sha256"]
    with pytest.raises(
        ExtractionContractError,
        match="missing expected value",
    ):
        load_and_validate_extraction_contract(
            checkpoint,
            tokenizer_vocabulary=VOCABULARY,
            expected_binding=incomplete,
        )
    with pytest.raises(
        ExtractionContractError,
        match="conflicting expected value",
    ):
        load_and_validate_extraction_contract(
            checkpoint,
            tokenizer_vocabulary=VOCABULARY,
            expected_binding=binding,
            expected_label_table_sha256="0" * 64,
        )


@pytest.mark.parametrize(
    "vocabulary",
    [
        {"": 0},
        {"a": True},
        {"a": -1},
        {"a": 0, "b": 0},
    ],
)
def test_tokenizer_vocabulary_is_strict(vocabulary, tmp_path: Path):
    checkpoint = _directory(tmp_path)
    with pytest.raises(ExtractionContractError, match="tokenizer"):
        publish_extraction_contract_once(
            checkpoint,
            tokenizer_vocabulary=vocabulary,
            entity_labels=ENTITY_LABELS,
            predicates=PREDICATES,
            schema_constraints=SCHEMA_CONSTRAINTS,
        )
    assert not (
        checkpoint / EXTRACTION_CONTRACT_FILENAME
    ).exists()


def test_atomic_create_once_publication_has_one_winner(tmp_path: Path):
    checkpoint = _directory(tmp_path)

    def attempt():
        try:
            return ("ok", _publish(checkpoint))
        except ExtractionContractError as exc:
            return ("error", str(exc))

    with ThreadPoolExecutor(max_workers=12) as executor:
        results = list(executor.map(lambda _: attempt(), range(24)))

    winners = [result for result in results if result[0] == "ok"]
    losers = [result for result in results if result[0] == "error"]
    assert len(winners) == 1
    assert len(losers) == 23
    assert all(
        "already exists" in result[1]
        or "create-once race" in result[1]
        for result in losers
    )
    assert (
        load_and_validate_extraction_contract(
            checkpoint,
            tokenizer_vocabulary=VOCABULARY,
            expected_binding=winners[0][1],
        )
        == winners[0][1]
    )
