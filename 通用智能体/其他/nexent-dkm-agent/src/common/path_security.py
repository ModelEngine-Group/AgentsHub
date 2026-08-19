"""Path validation helpers for local demo/API boundaries."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ALLOWED_ROOTS = (
    PROJECT_ROOT,
    Path(tempfile.gettempdir()),
)


def resolve_allowed_path(
    value: str | Path | None,
    *,
    allowed_roots: Iterable[str | Path] = DEFAULT_ALLOWED_ROOTS,
    label: str = "path",
) -> Path | None:
    """Resolve a user-supplied path and require it to stay inside allowed roots."""

    if value in (None, ""):
        return None

    candidate = Path(value).expanduser().resolve(strict=False)
    roots = tuple(Path(root).expanduser().resolve(strict=False) for root in allowed_roots)
    if any(_is_relative_to(candidate, root) for root in roots):
        return candidate
    allowed = ", ".join(str(root) for root in roots)
    raise ValueError(f"{label} is outside allowed roots: {candidate} (allowed: {allowed})")


def safe_path_string(value: str | Path | None, *, label: str = "path") -> str | None:
    """Return an allowed absolute path string for API/pipeline calls."""

    path = resolve_allowed_path(value, label=label)
    return str(path) if path is not None else None


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
