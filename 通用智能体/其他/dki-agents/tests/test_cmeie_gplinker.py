import pytest

torch = pytest.importorskip("torch")

from dki.nlp.cmeie_gplinker import (
    DirectionalGlobalPointer,
    RelationPairPointer,
    decode_spo_token_links,
    pointer_group_loss,
)


def test_relation_pointer_keeps_reverse_direction_link():
    pointer = RelationPairPointer(hidden_size=4, heads=1, inner_dim=2)
    logits = pointer(torch.ones(1, 5, 4), torch.tensor([[1, 1, 1, 1, 1]]))
    assert logits.shape == (1, 1, 5, 5)
    assert logits[0, 0, 3, 1] > -1e11
    assert logits[0, 0, 0, 1] < -1e11


def test_classic_directional_pointer_keeps_reverse_direction_link():
    pointer = DirectionalGlobalPointer(
        hidden_size=4,
        heads=2,
        inner_dim=2,
    )
    logits = pointer(
        torch.ones(1, 5, 4),
        torch.tensor([[1, 1, 1, 1, 1]]),
    )
    assert logits.shape == (1, 2, 5, 5)
    assert logits[0, 1, 3, 1] > -1e11
    assert logits[0, 1, 0, 1] < -1e11


def test_decode_requires_both_relation_links_and_entity_spans():
    entities = [(0, 1, 1), (1, 3, 3)]
    assert decode_spo_token_links(entities, [(2, 1, 3)], []) == set()
    assert decode_spo_token_links(entities, [(2, 1, 3)], [(2, 1, 3)]) == {(0, 1, 1, 2, 1, 3, 3)}


def test_pointer_group_loss_sums_heads_before_batch_mean():
    logits = torch.zeros((1, 1, 2, 2))
    labels = torch.zeros_like(logits)
    labels[0, 0, 0, 1] = 1
    one_head = pointer_group_loss(logits, labels)
    three_heads = pointer_group_loss(
        logits.repeat(1, 3, 1, 1),
        labels.repeat(1, 3, 1, 1),
    )
    assert torch.allclose(three_heads, one_head * 3)


def test_sparse_positive_gradient_is_not_diluted_by_empty_relation_heads():
    logits = torch.full((1, 44, 1, 1), -10.0, requires_grad=True)
    labels = torch.zeros_like(logits)
    labels[0, 0, 0, 0] = 1
    pointer_group_loss(logits, labels).backward()
    assert logits.grad is not None
    assert float(logits.grad[0, 0, 0, 0]) < -0.99


def test_global_head_reduction_balances_one_multilabel_space_per_sample():
    logits = torch.zeros((1, 1, 2, 2))
    labels = torch.zeros_like(logits)
    labels[0, 0, 0, 1] = 1
    one_head = pointer_group_loss(
        logits,
        labels,
        head_reduction="global",
    )
    three_heads = pointer_group_loss(
        logits.repeat(1, 3, 1, 1),
        labels.repeat(1, 3, 1, 1),
        head_reduction="global",
    )
    assert one_head < three_heads
    assert three_heads < one_head * 3


def test_balanced_head_reduction_is_invariant_to_repeated_empty_heads():
    logits = torch.zeros((1, 2, 2, 2))
    labels = torch.zeros_like(logits)
    labels[0, 0, 0, 1] = 1
    two_heads = pointer_group_loss(
        logits,
        labels,
        head_reduction="balanced",
    )
    repeated_logits = torch.cat(
        [logits[:, :1], logits[:, 1:].repeat(1, 9, 1, 1)],
        dim=1,
    )
    repeated_labels = torch.cat(
        [labels[:, :1], labels[:, 1:].repeat(1, 9, 1, 1)],
        dim=1,
    )
    ten_heads = pointer_group_loss(
        repeated_logits,
        repeated_labels,
        head_reduction="balanced",
    )
    assert torch.allclose(two_heads, ten_heads)


def test_positive_weight_scales_only_positive_logit_gradients():
    labels = torch.zeros((1, 2, 2, 2))
    labels[0, 0, 0, 1] = 1
    baseline_logits = torch.zeros_like(labels, requires_grad=True)
    weighted_logits = torch.zeros_like(labels, requires_grad=True)

    pointer_group_loss(
        baseline_logits,
        labels,
        head_reduction="balanced",
        positive_weight=1.0,
    ).backward()
    pointer_group_loss(
        weighted_logits,
        labels,
        head_reduction="balanced",
        positive_weight=4.0,
    ).backward()

    positive = labels.bool()
    assert torch.allclose(
        weighted_logits.grad[positive],
        baseline_logits.grad[positive] * 4.0,
    )
    assert torch.allclose(
        weighted_logits.grad[~positive],
        baseline_logits.grad[~positive],
    )


def test_active_margin_ignores_empty_heads_and_strengthens_positive_gradient():
    labels = torch.zeros((1, 2, 2, 2))
    labels[0, 0, 0, 1] = 1
    baseline_logits = torch.zeros_like(labels, requires_grad=True)
    margin_logits = torch.zeros_like(labels, requires_grad=True)

    pointer_group_loss(
        baseline_logits,
        labels,
        head_reduction="active_margin",
        positive_margin=0.0,
    ).backward()
    pointer_group_loss(
        margin_logits,
        labels,
        head_reduction="active_margin",
        positive_margin=3.0,
    ).backward()

    assert (
        abs(float(margin_logits.grad[0, 0, 0, 1]))
        > abs(float(baseline_logits.grad[0, 0, 0, 1]))
    )
    assert torch.count_nonzero(margin_logits.grad[:, 1:]) == 0

    repeated = pointer_group_loss(
        margin_logits.detach().repeat(1, 8, 1, 1),
        labels.repeat(1, 8, 1, 1),
        head_reduction="active_margin",
        positive_margin=3.0,
    )
    one = pointer_group_loss(
        margin_logits.detach()[:, :1],
        labels[:, :1],
        head_reduction="active_margin",
        positive_margin=3.0,
    )
    assert torch.allclose(repeated, one)


def test_balanced_head_reduction_averages_active_and_inactive():
    base_logits = torch.zeros((1, 2, 1, 1), requires_grad=True)
    base_labels = torch.zeros_like(base_logits)
    base_labels[0, 0, 0, 0] = 1
    base_loss = pointer_group_loss(
        base_logits,
        base_labels,
        head_reduction="balanced",
    )
    base_loss.backward()

    repeated_logits = torch.zeros((1, 10, 1, 1), requires_grad=True)
    repeated_labels = torch.zeros_like(repeated_logits)
    repeated_labels[0, 0, 0, 0] = 1
    repeated_loss = pointer_group_loss(
        repeated_logits,
        repeated_labels,
        head_reduction="balanced",
    )
    repeated_loss.backward()

    assert torch.allclose(base_loss, repeated_loss)
    assert torch.allclose(base_logits.grad[:, 0], repeated_logits.grad[:, 0])
    assert torch.allclose(
        base_logits.grad[:, 1],
        repeated_logits.grad[:, 1:].sum(dim=1),
    )
    assert torch.all(repeated_logits.grad[:, 1:] > 0)


def test_active_bce_balances_positive_and_negative_gradient_mass():
    labels = torch.zeros((1, 2, 2, 2))
    labels[0, 0, 0, 1] = 1
    logits = torch.zeros_like(labels, requires_grad=True)

    pointer_group_loss(
        logits,
        labels,
        head_reduction="active_bce",
    ).backward()

    positive = labels.bool()
    positive_mass = -logits.grad[positive].sum()
    negative_mass = logits.grad[~positive].sum()
    assert torch.allclose(positive_mass, negative_mass)
    assert torch.count_nonzero(logits.grad[:, 1:]) == 0


def test_balanced_bce_supervises_empty_heads_without_count_dilution():
    labels = torch.zeros((1, 2, 2, 2))
    labels[0, 0, 0, 1] = 1
    logits = torch.zeros_like(labels, requires_grad=True)
    pointer_group_loss(
        logits,
        labels,
        head_reduction="balanced_bce",
    ).backward()
    assert torch.count_nonzero(logits.grad[:, 1:]) > 0

    repeated_logits = torch.zeros(
        (1, 9, 2, 2),
        requires_grad=True,
    )
    repeated_labels = torch.zeros_like(repeated_logits)
    repeated_labels[0, 0, 0, 1] = 1
    two_heads = pointer_group_loss(
        logits.detach(),
        labels,
        head_reduction="balanced_bce",
    )
    nine_heads = pointer_group_loss(
        repeated_logits,
        repeated_labels,
        head_reduction="balanced_bce",
    )
    assert torch.allclose(two_heads, nine_heads)


def test_hierarchical_global_reduction_matches_flat_reference_and_gradient():
    torch.manual_seed(7)
    labels = (torch.rand((2, 4, 3, 3)) > 0.82).float()
    hierarchical_logits = torch.randn(
        (2, 4, 3, 3),
        requires_grad=True,
    )
    flat_logits = hierarchical_logits.detach().clone().requires_grad_(True)

    hierarchical = pointer_group_loss(
        hierarchical_logits,
        labels,
        head_reduction="global",
    )
    batch = flat_logits.shape[0]
    flat_labels = labels.reshape(batch, 1, -1)
    signed = (
        (1 - 2 * flat_labels)
        * flat_logits.reshape(batch, 1, -1)
    )
    negative = signed - flat_labels * 1e12
    positive = signed - (1 - flat_labels) * 1e12
    zero = torch.zeros_like(signed[..., :1])
    reference = (
        torch.logsumexp(
            torch.cat([negative, zero], dim=-1),
            dim=-1,
        )
        + torch.logsumexp(
            torch.cat([positive, zero], dim=-1),
            dim=-1,
        )
    ).mean()

    hierarchical.backward()
    reference.backward()
    assert torch.allclose(hierarchical, reference, atol=1e-6)
    assert torch.allclose(
        hierarchical_logits.grad,
        flat_logits.grad,
        atol=1e-6,
    )


def test_pointer_group_loss_rejects_unknown_reduction():
    with pytest.raises(ValueError, match="unsupported"):
        pointer_group_loss(
            torch.zeros((1, 1, 1, 1)),
            torch.zeros((1, 1, 1, 1)),
            head_reduction="unknown",
        )


@pytest.mark.parametrize("positive_weight", [0.0, -1.0, float("inf")])
def test_pointer_group_loss_rejects_invalid_positive_weight(
    positive_weight,
):
    with pytest.raises(ValueError, match="positive_weight"):
        pointer_group_loss(
            torch.zeros((1, 1, 1, 1)),
            torch.zeros((1, 1, 1, 1)),
            positive_weight=positive_weight,
        )


@pytest.mark.parametrize("positive_margin", [-1.0, float("inf")])
def test_pointer_group_loss_rejects_invalid_positive_margin(
    positive_margin,
):
    with pytest.raises(ValueError, match="positive_margin"):
        pointer_group_loss(
            torch.zeros((1, 1, 1, 1)),
            torch.zeros((1, 1, 1, 1)),
            positive_margin=positive_margin,
        )


def test_npu_autocast_when_npu_is_available():
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("Ascend NPU unavailable in this environment")
    tensor = torch.ones((2, 2), device="npu:0")
    with torch.autocast(device_type="npu", dtype=torch.float16):
        result = tensor @ tensor
    assert result.device.type == "npu"
