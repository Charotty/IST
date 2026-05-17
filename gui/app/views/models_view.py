"""Модели — метаданные обученного bundle (справочная вкладка)."""

from __future__ import annotations

from typing import Optional

from PyQt6.QtWidgets import (
    QLabel,
    QPlainTextEdit,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from gui.api import IstGuiClient
from gui.api.types import BundleInfo
from gui.app.widgets.help_label import help_label
from gui.app.workers import BundleInfoWorker


class ModelsView(QWidget):
    def __init__(self, api: Optional[IstGuiClient] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._api = api or IstGuiClient()
        self._symbol = "BTC/USDT"
        self._timeframe = "1h"
        self._worker: Optional[BundleInfoWorker] = None

        layout = QVBoxLayout(self)

        about = help_label(
            "<b>Зачем эта вкладка?</b><br>"
            "Здесь не обучение, а <i>паспорт</i> уже сохранённой модели (artifact bundle) "
            "для выбранной пары и таймфрейма: какие алгоритмы входят в ансамбль, "
            "какие признаки ожидаются на входе, какие веса trend/range зашиты в конфиг.<br><br>"
            "Нужна для отладки и отчёта: убедиться, что GUI/CLI смотрят на тот же run_id, "
            "что схема признаков совпадает с parquet, и понять состав 4 моделей (lgb, gru, xgb, cnn). "
            "Обучение — вкладка «Задачи» или CLI prepare-symbol / train-final.",
            background="#e3f2fd",
            border="#90caf9",
            padding="12px",
        )
        layout.addWidget(about)

        self._summary = QLabel("Загрузка метаданных бандла…")
        self._summary.setWordWrap(True)
        self._summary.setStyleSheet("color: #000000;")
        layout.addWidget(self._summary)

        tabs = QTabWidget()
        self._features = QPlainTextEdit()
        self._features.setReadOnly(True)
        tabs.addTab(self._features, "Список признаков")

        self._orch = QPlainTextEdit()
        self._orch.setReadOnly(True)
        tabs.addTab(self._orch, "Веса ансамбля (trend / range)")

        layout.addWidget(tabs, stretch=1)

    def set_context(self, symbol: str, timeframe: str) -> None:
        self._symbol = symbol
        self._timeframe = timeframe

    def refresh(self) -> None:
        if self._worker and self._worker.isRunning():
            return
        self._summary.setText("Загрузка…")
        self._worker = BundleInfoWorker(self._symbol, self._timeframe, api=self._api)
        self._worker.finished.connect(self._on_bundle)
        self._worker.failed.connect(self._on_fail)
        self._worker.start()

    def _on_fail(self, msg: str) -> None:
        self._summary.setText(
            f"Bundle не найден для {self._symbol} {self._timeframe}.\n"
            f"{msg}\n\n"
            "Обучите модель: вкладка «Задачи» → «Финальное обучение» "
            "или CLI train-final-symbol."
        )
        self._features.clear()
        self._orch.clear()

    def _on_bundle(self, info: BundleInfo) -> None:
        if info.schema_valid is True:
            schema = "совпадает"
        elif info.schema_valid is False:
            schema = "не совпадает"
        else:
            schema = "не проверялась"
        self._summary.setText(
            f'<div style="color:#000000;"><b>Прогон:</b> {info.run_id}<br>'
            f"<b>Создан:</b> {info.created_at}<br>"
            f"<b>Модели:</b> {', '.join(info.model_keys)}<br>"
            f"<b>Схема признаков:</b> {info.feature_schema_hash} ({schema})<br>"
            f"<b>Порог meta (train):</b> {info.train_meta_threshold}</div>"
        )
        self._summary.setStyleSheet("color: #000000;")
        self._features.setPlainText("\n".join(info.feature_columns))
        import json

        orch = info.orchestrator_config
        text = json.dumps(
            {
                "trend_weights": orch.get("trend_weights", {}),
                "range_weights": orch.get("range_weights", {}),
                "ensemble_mode": orch.get("ensemble_mode"),
            },
            indent=2,
            ensure_ascii=False,
        )
        self._orch.setPlainText(text)
