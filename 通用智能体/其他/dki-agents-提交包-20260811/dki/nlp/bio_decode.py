"""Reference BIO decoder shared by PyTorch and OM inference paths."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


class BioDecoder:
    """Decode BIO labels using a canonical, deterministic state machine.

    Malformed ``I-X`` starts a new X entity.  ``I-Y`` following X closes X and
    starts Y.  Invalid/special positions close the active entity.
    """

    def __init__(self, id2label: Dict[int, str]):
        self.id2label = {int(key): str(value) for key, value in id2label.items()}

    def decode_typed(self, pred_ids: np.ndarray, offsets: np.ndarray) -> List[List[dict]]:
        if pred_ids.ndim != 2 or offsets.shape != pred_ids.shape + (2,):
            raise ValueError(
                f"pred_ids/offsets shape mismatch: {pred_ids.shape}, {offsets.shape}"
            )
        output: List[List[dict]] = []
        for row_index in range(pred_ids.shape[0]):
            entities: List[dict] = []
            active_type = None
            active_start = active_end = active_col = None

            def close() -> None:
                nonlocal active_type, active_start, active_end, active_col
                if active_type is not None:
                    entities.append({
                        "type": active_type,
                        "char_start": int(active_start),
                        "char_end": int(active_end),
                        "b_col": int(active_col),
                    })
                active_type = active_start = active_end = active_col = None

            for col in range(pred_ids.shape[1]):
                start, end = (int(value) for value in offsets[row_index, col])
                label = self.id2label.get(int(pred_ids[row_index, col]), "O")
                valid = end > start
                prefix, entity_type = (label.split("-", 1) + [""])[:2] if "-" in label else (label, "")
                if not valid or prefix not in {"B", "I"} or not entity_type:
                    close()
                    continue
                if prefix == "B" or active_type != entity_type:
                    close()
                    active_type = entity_type
                    active_start = start
                    active_end = end
                    active_col = col
                else:
                    active_end = end
            close()
            output.append(entities)
        return output

    def decode(self, pred_ids: np.ndarray, offsets: np.ndarray) -> Tuple[int, np.ndarray]:
        typed = self.decode_typed(pred_ids, offsets)
        spans = [
            (row, entity["b_col"], entity["char_start"], entity["char_end"])
            for row, entities in enumerate(typed)
            for entity in entities
        ]
        return len(spans), np.asarray(spans, dtype=np.int64).reshape((-1, 4))
