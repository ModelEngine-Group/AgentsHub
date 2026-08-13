from dki.nlp.windowing import (
    WindowingError,
    choose_shape,
    merge_window_entities,
    plan_token_windows,
)


def _offsets(n):
    return [(index, index + 1) for index in range(n)]


def test_shape_selection_is_smallest_fit():
    assert choose_shape(60, [256, 64, 128]) == 64
    assert choose_shape(63, [64, 128]) == 128
    assert choose_shape(300, [64, 128, 256]) is None


def test_windows_cover_long_text_with_cores():
    windows = plan_token_windows(_offsets(513), capacity=254, overlap=32)
    assert windows[0].token_start == 0
    assert windows[-1].token_end == 513
    owned = [token for token in range(513) if sum(w.core_start <= token < w.core_end for w in windows) == 1]
    assert owned == list(range(513))


def test_merge_is_order_independent_and_position_preserving():
    windows = plan_token_windows(_offsets(300), capacity=254, overlap=32)
    first = {"span": [220, 224], "type": "Disease", "mention": "同名", "anchor_token": 220}
    duplicate = dict(first)
    second = {"span": [280, 284], "type": "Disease", "mention": "同名", "anchor_token": 280}
    values = [(windows[0], [first]), (windows[1], [duplicate, second])]
    forward = merge_window_entities(values)
    reverse = merge_window_entities(reversed(values))
    assert forward == reverse
    assert [item["span"] for item in forward] == [[220, 224], [280, 284]]


def test_entity_outside_window_fails_closed():
    window = plan_token_windows(_offsets(10), capacity=8, overlap=2)[0]
    try:
        merge_window_entities([(window, [{"span": [0, 9], "type": "Exam", "anchor_token": 1}])])
    except WindowingError:
        pass
    else:
        raise AssertionError("expected WindowingError")


def test_invalid_offsets_rejected():
    try:
        plan_token_windows([(1, 2), (0, 1)], capacity=4, overlap=1)
    except WindowingError:
        pass
    else:
        raise AssertionError("expected WindowingError")
