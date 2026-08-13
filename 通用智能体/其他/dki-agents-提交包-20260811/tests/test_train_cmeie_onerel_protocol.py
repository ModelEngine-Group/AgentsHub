from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "train_cmeie_onerel.py"


def test_formal_recipe_is_fixed_and_has_no_outer_or_official_dev_cli():
    source = SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source)
    model_source = (ROOT / "dki" / "nlp" / "cmeie_onerel.py").read_text(
        encoding="utf-8"
    )
    assert "ONEREL_ARCHITECTURE_VERSION = 4" in model_source
    assert "sys.path.insert(0, str(ROOT))" in source
    assert "self.proposal_calibrator = nn.Linear(3, 1)" in model_source
    assert "nn.init.zeros_(self.proposal_calibrator.weight)" in model_source
    assert "self.verifier_context_projection = nn.Sequential(" in model_source
    assert "nn.init.zeros_(self.verifier_context_output.weight)" in model_source
    assert "FORMAL_MAX_LENGTH = 320" in source
    assert "FORMAL_PAIR_RANK = 192" in source
    assert "FORMAL_LORA_RANK = 8" in source
    assert "FORMAL_HARD_NEGATIVE_RATIO = 16" in source
    assert "ONLINE_MINING_THRESHOLD = 0.0" in source
    assert "FIXED_PROPOSAL_REFINEMENT_THRESHOLD = -0.5" in source
    assert "FORMAL_REFINEMENT_THRESHOLD = -0.5" in source
    assert "REFINEMENT_HARD_NEGATIVE_RATIO = 8" in source
    assert "REFINEMENT_RAW_CANDIDATE_CAP = 100_000" in source
    assert "REFINEMENT_SELECTED_CANDIDATE_CAP = 4096" in source
    assert '"candidate-verifier"' in source
    assert 'parser.add_argument("--calibrator-lr", type=float, default=1e-2)' in source
    assert "OVERFIT_REFINEMENT_STEPS = 100" in source
    assert source.count("batch_size=len(rows)") == 2
    assert "batches = [list(rows) for _ in range(maximum_optimizer_steps)]" in source
    assert "catastrophic row forgetting" in source
    assert "encoder_base_frozen" in source
    assert "all_bert_query_and_value" in source
    assert "online_verifier_hard_negatives" in source
    assert "online_candidate_cap_per_row\": 4096" in source
    assert "online_mining_threshold\": ONLINE_MINING_THRESHOLD" in source
    assert "online_mining_initial_warmup_optimizer_steps\": 256" in source
    assert 'formal_online_mining\": "disabled_after_density_ablation"' in source
    assert "online_warmup_optimizer_steps=256" in source
    assert "online_mining_enabled=False" in source
    assert "if epoch == args.epochs" in source
    assert "proposal_threshold=FORMAL_REFINEMENT_THRESHOLD" in source
    assert "verifier_separation_margin\": 1.0" in source
    assert "all_positive_x_top_hard_negative_pairwise" in source
    assert '"positive_floor": 1.0' in source
    assert '"negative_ceiling": -1.0' in source
    assert '"tail_count_per_row": 8' in source
    assert "verifier_gradient_to_shared_hidden\": False" in source
    assert "verifier_dropout\": 0.1" in source
    assert 'link_hard_negative_scope\": "per_source_row"' in source
    assert '"proposal_and_encoder_frozen": True' in source
    assert '"negative_selection": "proposal_score_hardest_then_coordinate"' in source
    assert '"proposal_score_features": ["relation", "link_min", "boundary_min"]' in source
    assert '"all_gold_retained": args.refine_scope == "verifier"' in source
    assert 'elif args.refine_scope == "candidate-verifier"' in source
    assert "active_refinement_parameters = semantic_verifier_parameters" in source
    assert "retain_unproposed_gold = False" in source
    assert "train_verifier_dropout = False" in source
    assert "refinement_weight_decay = 0.0" in source
    assert '"inference_truncation": False' in source
    assert '"formal_refinement_epochs"' in source
    assert "maximum_span_candidates=maximum_raw_candidates_per_row" in source
    assert "select_fixed_proposal_candidates(" in source
    assert 'choices=("overfit", "train", "oof-train", "refine")' in source
    assert '"fit_internal_oof_fold_selection"' in source
    assert '"oof_candidate_generator_completed": is_oof_training' in source
    assert "OOF proposal training must not depend on an exact-SPO overfit receipt" in source
    assert "in-sample fit proposals are forbidden" in (
        ROOT / "scripts" / "train_cmeie_candidate_cross_encoder.py"
    ).read_text(encoding="utf-8")
    assert "low_threshold_verifier_refinement" in source
    assert "model.score_triples(\n                hidden.detach()," in source
    assert '"proposal_head": 1.0' in source
    assert '"verifier": 1.0' in source
    assert "verifier_ids" in source
    assert any(
        isinstance(node, ast.FunctionDef) and node.name == "_online_verifier_batch"
        for node in tree.body
    )
    assert any(
        isinstance(node, ast.FunctionDef)
        and node.name == "_fixed_proposal_verifier_batch"
        for node in tree.body
    )
    assert any(
        isinstance(node, ast.FunctionDef)
        and node.name == "_refine_verifier_on_fixed_proposals"
        for node in tree.body
    )
    functions = {
        node.name: ast.unparse(node)
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }
    assert "batches = [list(rows) for _ in range(maximum_optimizer_steps)]" in functions[
        "_refine_verifier_on_fixed_proposals"
    ]
    assert "cached_fixed_payload" in functions["_refine_verifier_on_fixed_proposals"]
    assert "catastrophic row forgetting" not in functions["_train"]
    assert "--outer" not in source
    assert "--official-dev" not in source
    assert any(isinstance(node, ast.FunctionDef) and node.name == "_write_json_once" for node in tree.body)


def test_calibration_grid_is_global_not_per_predicate():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "THRESHOLDS = (-0.5, 0.0, 0.5)" in source
    assert "relation\": THRESHOLDS" in source
    assert "link\": THRESHOLDS" in source
    assert "boundary\": THRESHOLDS" in source
    assert "verifier\": THRESHOLDS" in source
    assert "for verifier in THRESHOLDS" in source
    assert "per_predicate_threshold" not in source
