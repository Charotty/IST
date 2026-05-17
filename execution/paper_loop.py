"""
Один шаг paper/live: окно признаков → InferenceOrchestrator → ExecutionManager.

Внешний цикл по закрытию бара остаётся у вызывающего кода (биржа / планировщик).
"""

from __future__ import annotations

from typing import Any, Optional
import logging

import pandas as pd


def run_inference_execution_step(
    orchestrator: Any,
    features_window: pd.DataFrame,
    execution_manager: Any,
    symbol: str,
    current_price: float,
    rl_risk_multiplier: float = 1.0,
) -> Optional[Any]:
    """
    Вызывает ``orchestrator.predict(features_window)`` и передаёт результат в ``execute_signal``.

    Args:
        orchestrator: экземпляр ``InferenceOrchestrator`` (после ``initialize``).
        features_window: последние N баров признаков (как в ``batch_predict``).
        execution_manager: ``ExecutionManager``.
        symbol: тикер для брокера.
        current_price: цена исполнения (например close бара).
        rl_risk_multiplier: множитель из ``rl_layer`` (1.0 если выкл.).
    """
    log = logging.getLogger(__name__)
    if not getattr(orchestrator, "is_initialized", False):
        raise ValueError("orchestrator must be initialized")

    inf = orchestrator.predict(features_window)
    signal = int(inf.signal)
    base_size = float(inf.position_size)
    if base_size == 0.0 and signal != 0:
        base_size = 1.0

    order = execution_manager.execute_signal(
        signal=signal,
        symbol=symbol,
        position_size=base_size,
        current_price=current_price,
        risk_multiplier=rl_risk_multiplier,
    )
    if order is None:
        log.info("No order (signal=%s)", signal)
    return order
