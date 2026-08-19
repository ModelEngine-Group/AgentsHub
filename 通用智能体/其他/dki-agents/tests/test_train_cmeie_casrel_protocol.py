import pytest

from scripts.train_cmeie_casrel import _parser, main


def test_formal_casrel_defaults_are_the_stable_top2_recipe():
    args = _parser().parse_args(
        [
            "--mode",
            "train",
            "--run-root",
            "run",
            "--fit-data",
            "fit.jsonl",
            "--split-manifest",
            "split.json",
            "--base-model",
            "model",
            "--output",
            "out",
            "--source-tree-sha256",
            "a" * 64,
        ]
    )

    assert args.epochs == 6
    assert args.head_only_epochs == 4
    assert args.joint_top_encoder_layers == 2
    assert args.joint_encoder_lr == 1e-6
    assert args.joint_encoder_adam_eps == 1e-4
    assert args.head_lr == 5e-5
    assert args.joint_head_lr == 5e-5


def test_formal_casrel_refuses_full_unfreeze_before_data_access(tmp_path):
    args = [
        "--mode",
        "train",
        "--run-root",
        str(tmp_path / "run"),
        "--fit-data",
        str(tmp_path / "fit.jsonl"),
        "--split-manifest",
        str(tmp_path / "split.json"),
        "--base-model",
        str(tmp_path / "model"),
        "--output",
        str(tmp_path / "out"),
        "--source-tree-sha256",
        "a" * 64,
        "--joint-top-encoder-layers",
        "0",
    ]

    with pytest.raises(SystemExit, match="top-2"):
        main(args)
