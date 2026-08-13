"""Static protocol checks for the predicate-span calibration trainer."""

from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "train_cmeie_predicate_span.py"
)


def test_trainer_has_no_outer_or_official_dev_input():
    text = SCRIPT.read_text(encoding="utf-8")
    assert 'parser.add_argument("--fit-data"' in text
    assert 'parser.add_argument("--calibration-data"' in text
    assert 'parser.add_argument("--outer' not in text
    assert 'parser.add_argument("--official' not in text
    assert '"outer_accessed": False' in text
    assert '"official_dev_accessed": False' in text


def test_trainer_uses_declared_global_grid_and_create_once_publication():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "DECLARED_THRESHOLD_VALUES = (-0.5, 0.0, 0.5)" in text
    assert "_publish_directory_create_once(staging, args.output_dir)" in text
    assert '"effective_batch_size"' in text
    assert '"source_tree_manifest_sha256"' in text
    assert "pad_to_multiple_of=16" in text
    assert "_length_bucket_rows(" in text
    assert 'parser.add_argument("--maximum-active-candidates"' in text
    assert 'parser.add_argument("--maximum-join-work"' in text
    assert '"selected_maximum_predictions_per_example"' in text
    assert '"grid_maximum_predictions_per_example"' in text
    assert '"selected_maximum_join_work_per_example"' in text
    assert '"grid_maximum_join_work_per_example"' in text
    assert '"join_work_rejected_combinations"' in text
    assert '"declared_threshold_combination_count"' in text
    assert '"density_rejected_combinations"' in text
    assert '"density_rejected_thresholds"' in text
    assert '"remaining_threshold_combination_count"' in text
    assert "**density_report" in text
    assert "**best_density_report" in text


def test_trainer_allows_fully_frozen_encoder_pilot():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "0 <= args.head_warmup_epochs <= args.epochs" in text
    assert '"head-warmup-epochs must not exceed epochs"' in text
    assert "_set_training_mode(model, encoder_trainable=encoder_trainable)" in text
    assert "model.encoder.eval()" in text


def test_high_threshold_grid_is_explicitly_marked_as_ablation():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "HIGH_THRESHOLD_ABLATION_VALUES = (0.0, 0.5, 1.0)" in text
    assert '"--threshold-grid-mode"' in text
    assert '"evidence_driven_ablation"' in text
    assert '"threshold_grid_mode"' in text
    assert '"available_threshold_grid_for_mode"' in text


def test_balanced_sparse_loss_is_explicitly_bound_as_an_ablation():
    text = SCRIPT.read_text(encoding="utf-8")
    assert '"--loss-mode"' in text
    assert "choices=PREDICATE_SPAN_LOSS_MODES" in text
    assert "default=LOSS_MODE_HARD_NEGATIVE_BCE" in text
    assert "loss_mode=args.loss_mode" in text
    assert '"loss_mode": args.loss_mode' in text
    assert '"loss_ablation": loss_ablation' in text
    assert '"balanced_per_head_sparse_logsumexp_ablation"' in text
    assert '"promotion_eligible": passed and not loss_ablation' in text
    assert '"loss_ablation_quality_gate_passed"' in text


def test_fit_only_smoke_binds_the_same_explicit_loss_ablation():
    smoke = SCRIPT.with_name("smoke_cmeie_predicate_span.py")
    text = smoke.read_text(encoding="utf-8")

    assert '"--loss-mode"' in text
    assert "choices=PREDICATE_SPAN_LOSS_MODES" in text
    assert "default=LOSS_MODE_HARD_NEGATIVE_BCE" in text
    assert "loss_mode=args.loss_mode" in text
    assert '"loss_mode": args.loss_mode' in text
    assert '"loss_ablation": args.loss_mode != LOSS_MODE_HARD_NEGATIVE_BCE' in text
    assert '"negatives_per_head_applied": args.loss_mode == LOSS_MODE_HARD_NEGATIVE_BCE' in text


def test_trainer_binds_deterministic_cross_sentence_policy():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "allow_cross_sentence_fallback=True" in text
    assert '"cross_sentence_pair_policy"' in text
    assert '"same_sentence_or_unique_deterministic_nearest_fallback"' in text


def test_failed_quality_gate_is_published_only_as_failed_evidence():
    text = SCRIPT.read_text(encoding="utf-8")
    assert '"failed_evidence"' in text
    assert 'staging / "FAILED_EVIDENCE.json"' in text
    assert "return 0 if passed else 2" in text


def test_checkpoint_rescore_is_declared_grid_only_and_never_promotable():
    text = SCRIPT.read_text(encoding="utf-8")

    assert '"--rescore-checkpoint-dir"' in text
    assert "checkpoint rescoring requires the exact declared threshold grid" in text
    assert '"posthoc_declared_grid_rescore": True' in text
    assert '"threshold_grid_role": "posthoc_declared_grid_rescore"' in text
    assert '"training_performed": False' in text
    assert '"promotion_eligible": False' in text
    assert '"checkpoint_model_state_sha256"' in text
    assert '"checkpoint_candidate_receipt_sha256"' in text
    assert '"evaluator_sha256"' in text
    assert 'staging / "rescore_receipt.json"' in text
