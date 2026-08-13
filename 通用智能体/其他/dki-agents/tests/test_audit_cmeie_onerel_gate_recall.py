from scripts.audit_cmeie_onerel_gate_recall import (
    gate_key,
    gate_passes,
    summarize_gate_recall,
)


def test_gate_passes_can_disable_non_onerel_auxiliary_gates():
    scores = (0.7, -4.0, -3.0)
    assert gate_passes(
        scores,
        link_threshold=0.5,
        relation_threshold=None,
        boundary_threshold=None,
    )
    assert not gate_passes(
        scores,
        link_threshold=0.5,
        relation_threshold=-0.5,
        boundary_threshold=None,
    )
    assert gate_key(0.5, None, None) == "link=0.5|relation=off|boundary=off"


def test_gate_recall_is_surface_level_and_uses_any_aligned_mention():
    rows = [
        (
            "row-1",
            {
                ("甲", "病因", "乙"): [
                    (-1.0, 2.0, 2.0),
                    (0.75, -2.0, -2.0),
                ],
                ("丙", "治疗", "丁"): [],
            },
        )
    ]
    result = summarize_gate_recall(rows)
    link_only = result["grid"]["link=0.5|relation=off|boundary=off"]
    fully_gated = result["grid"]["link=0.5|relation=-0.5|boundary=-0.5"]
    assert result["surface_gold"] == 2
    assert result["unaligned_surface_gold"] == 1
    assert link_only["covered"] == 1
    assert link_only["recall"] == 0.5
    assert fully_gated["covered"] == 0
    assert result["grid_size"] == 48


def test_gate_recall_reports_per_predicate_link_ceiling():
    rows = [
        (
            "row-1",
            {
                ("甲", "病因", "乙"): [(0.1, -9.0, -9.0)],
                ("丙", "病因", "丁"): [(0.9, -9.0, -9.0)],
            },
        )
    ]
    result = summarize_gate_recall(rows)
    predicate = result["link_only_per_predicate"]["0.5"]["病因"]
    assert predicate == {"covered": 1, "total": 2, "recall": 0.5}
