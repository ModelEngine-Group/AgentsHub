from kg.graph_query import SAFETY_SUFFIX, answer_with_safety, dedupe_preserve, entity_label, limit_paths


def test_dedupe_preserves_order() -> None:
    assert dedupe_preserve(["a", "b", "a", "", "c"]) == ["a", "b", "c"]


def test_limit_paths_deduplicates_before_limiting() -> None:
    assert limit_paths(["a", "a", "b", "c"], limit=2) == ["a", "b"]


def test_answer_includes_safety_suffix() -> None:
    assert answer_with_safety("查询完成").endswith(SAFETY_SUFFIX)


def test_entity_label_prefers_display_name() -> None:
    assert entity_label({"id": "1", "name": "内部名", "display_name": "展示名"}) == "展示名"


def test_entity_label_falls_back_to_id() -> None:
    assert entity_label({"id": 123}) == "123"
