from __future__ import annotations

import os
from pathlib import Path
from typing import Union

_PathLike = Union[str, os.PathLike]

ENV_OUTPUT_ROOT = "EGB_OUTPUT_ROOT"
ENV_SCRATCH_ROOT = "SCRATCH_DIR"
DEFAULT_SUBDIR = "entropy-gated-branching"


def _coerce_path(value: _PathLike) -> Path:
    return Path(value).expanduser()


def get_output_root() -> Path:
    """Return the root directory for runtime outputs."""
    env_override = os.environ.get(ENV_OUTPUT_ROOT)
    if env_override:
        base = _coerce_path(env_override)
    else:
        scratch_base = os.environ.get(ENV_SCRATCH_ROOT)
        if scratch_base:
            base = _coerce_path(scratch_base)
        else:
            base = Path.home() / "links" / "scratch"
        base = base / DEFAULT_SUBDIR
    return base


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def scratch_subdir(*parts: _PathLike, create: bool = True) -> Path:
    target = get_output_root().joinpath(*(Path(p) if isinstance(p, os.PathLike) else p for p in parts))
    if create:
        ensure_dir(target)
    return target


def resolve_output_path(user_value: _PathLike | None, default_leaf: str) -> Path:
    root = get_output_root()
    if user_value is not None:
        candidate = _coerce_path(user_value)
        if not candidate.is_absolute():
            candidate = root / candidate
    else:
        candidate = root / default_leaf
    return ensure_dir(candidate)


def resolve_in_output_root(value: _PathLike) -> Path:
    candidate = _coerce_path(value)
    if not candidate.is_absolute():
        candidate = get_output_root() / candidate
    return candidate
