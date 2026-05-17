"""Runtime checks for ML dependencies (TensorFlow for GRU/CNN)."""

from __future__ import annotations

import sys
from typing import Iterable, Sequence


def needs_tensorflow(model_keys: Iterable[str]) -> bool:
    return any(k in ("gru", "cnn") for k in model_keys)


def require_tensorflow(model_keys: Sequence[str] | None = None) -> None:
    """
    Raise ``RuntimeError`` with a clear message if TensorFlow is not importable.

    GRU/CNN models import ``tensorflow`` at module load time; Keras in venv alone is not enough.
    """
    if model_keys is not None and not needs_tensorflow(model_keys):
        return
    try:
        import tensorflow as tf  # noqa: F401
    except ImportError as exc:
        keys = list(model_keys) if model_keys else ["gru", "cnn"]
        raise RuntimeError(
            f"TensorFlow is required for model_keys={keys} but cannot be imported.\n"
            f"  Python: {sys.executable}\n"
            "  Install in the **active** environment (the one shown above):\n"
            "    pip install \"tensorflow>=2.15,<2.20\"\n"
            "  If TensorFlow works with `py -3` but not `python`, you are using a different interpreter "
            "(e.g. system Python vs .venv). Activate .venv and install there."
        ) from exc
