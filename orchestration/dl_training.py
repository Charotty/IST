"""
TensorFlow training helpers for thesis acceleration (mixed precision, VRAM cleanup).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

_mixed_precision_enabled: Optional[bool] = None
_tf_runtime_configured = False


def configure_tf_runtime() -> None:
    """
    Call before first TensorFlow import in a process (tune-thesis entry).

    Reduces log spam; optional deterministic cuDNN (IST_TF_CUDNN_DETERMINISTIC=1).
    """
    global _tf_runtime_configured
    if _tf_runtime_configured:
        return
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    # Off by default: with TF_DETERMINISTIC_OPS=1 Keras requires tf.random.set_seed().
    if os.environ.get("IST_TF_CUDNN_DETERMINISTIC", "0").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")
        os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    _tf_runtime_configured = True


def _ensure_tf_random_seed() -> None:
    try:
        import tensorflow as tf

        seed = int(os.environ.get("IST_TF_SEED", "42"))
        tf.random.set_seed(seed)
    except Exception:
        pass


@dataclass
class DLTrainingProfile:
    mixed_precision: bool = False
    batch_size: int = 64
    epochs: int = 3
    use_xla: bool = False


def setup_mixed_precision(enable: bool) -> None:
    """Set global Keras mixed_float16 policy when GPU is available."""
    global _mixed_precision_enabled
    configure_tf_runtime()
    if os.environ.get("IST_DL_MIXED_PRECISION", "").strip().lower() in ("0", "false", "no"):
        enable = False
    if not enable:
        try:
            import tensorflow as tf

            tf.keras.mixed_precision.set_global_policy("float32")
        except Exception:
            pass
        _mixed_precision_enabled = False
        return
    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            _mixed_precision_enabled = True
        else:
            tf.keras.mixed_precision.set_global_policy("float32")
            _mixed_precision_enabled = False
    except Exception:
        _mixed_precision_enabled = False


def clear_tf_session() -> None:
    try:
        import tensorflow as tf

        tf.keras.backend.clear_session()
    except Exception:
        pass


def mixed_precision_active() -> bool:
    return bool(_mixed_precision_enabled)


def fit_dl_model(
    model: Any,
    X_train: Any,
    y_train: Any,
    *,
    validation_data: Optional[tuple] = None,
    profile: Optional[DLTrainingProfile] = None,
    callbacks: Optional[list] = None,
    verbose: int = 0,
) -> Any:
    """Fit Keras model with optional mixed precision policy."""
    prof = profile or DLTrainingProfile()
    setup_mixed_precision(prof.mixed_precision)
    _ensure_tf_random_seed()
    fit_kw: dict = {
        "epochs": prof.epochs,
        "batch_size": prof.batch_size,
        "callbacks": callbacks or [],
        "verbose": verbose,
    }
    if validation_data is not None:
        fit_kw["validation_data"] = validation_data
    if prof.use_xla:
        fit_kw["jit_compile"] = True
    return model.fit(X_train, y_train, **fit_kw)
