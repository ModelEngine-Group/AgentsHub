"""DataMate-style adapters for demo operators.

DataMate operators usually process one ``sample`` dictionary and return the
updated dictionary. This module keeps the demo operator unchanged and exposes a
small adapter that matches that style.
"""

from __future__ import annotations

from collections.abc import Mapping

from operators import CleanTextOperator


class CleanTextDataMateMapper:
    """Wrap ``CleanTextOperator`` as a DataMate-style ``sample -> sample`` mapper."""

    def __init__(self, remove_digits: bool = False, remove_punctuation: bool = False) -> None:
        self.remove_digits: bool = remove_digits
        self.remove_punctuation: bool = remove_punctuation
        self.operator: CleanTextOperator = CleanTextOperator()

    def execute(self, sample: Mapping[str, object]) -> dict[str, object]:
        if "text" not in sample:
            raise KeyError("DataMate sample must contain a 'text' field")

        result = dict(sample)
        result["text"] = self.operator.run(
            str(sample["text"]),
            remove_digits=self.remove_digits,
            remove_punctuation=self.remove_punctuation,
        )
        return result
