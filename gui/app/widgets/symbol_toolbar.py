"""Панель: пара OKX + фиксированные таймфреймы."""

from __future__ import annotations

from typing import List, Optional

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QWidget,
)

from gui.api.okx_api import standard_timeframes
from gui.api.types import SymbolEntry


class SymbolToolbar(QWidget):
    symbol_changed = pyqtSignal(str, str)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._markets: List[str] = []

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("Пара (OKX):"))
        self._symbol_combo = QComboBox()
        self._symbol_combo.setMinimumWidth(160)
        self._symbol_combo.setEditable(True)
        self._symbol_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        layout.addWidget(self._symbol_combo)

        layout.addWidget(QLabel("ТФ:"))
        self._tf_combo = QComboBox()
        self._tf_combo.addItems(standard_timeframes())
        self._tf_combo.setCurrentText("1h")
        layout.addWidget(self._tf_combo)

        self._bundle_label = QLabel("Модель: —")
        self._bundle_label.setStyleSheet("color: #666;")
        layout.addWidget(self._bundle_label)

        self._status_label = QLabel("●")
        self._status_label.setToolTip("Статус artifact bundle (локально)")
        layout.addWidget(self._status_label)

        layout.addStretch()

        self._refresh_btn = QPushButton("Обновить")
        layout.addWidget(self._refresh_btn)

        self._symbol_combo.currentIndexChanged.connect(self._emit_context)
        self._tf_combo.currentTextChanged.connect(self._emit_context)
        self._refresh_btn.clicked.connect(self._emit_context)

    def set_markets(self, symbols: List[str], *, select: Optional[str] = None) -> None:
        self._markets = list(symbols)
        self._symbol_combo.blockSignals(True)
        self._symbol_combo.clear()
        for s in symbols:
            self._symbol_combo.addItem(s, userData=s)
        pick = select or ("BTC/USDT" if "BTC/USDT" in symbols else (symbols[0] if symbols else ""))
        if pick:
            idx = self._symbol_combo.findData(pick)
            if idx >= 0:
                self._symbol_combo.setCurrentIndex(idx)
        self._symbol_combo.blockSignals(False)
        if symbols:
            self._emit_context()

    def set_pipeline_status(self, entry: Optional[SymbolEntry]) -> None:
        if entry and entry.has_bundle and entry.latest_bundle_run_id:
            self._bundle_label.setText(f"Модель: {entry.latest_bundle_run_id[:22]}…")
            self._status_label.setStyleSheet("color: #2e7d32; font-weight: bold;")
            self._status_label.setToolTip("Обученная модель (bundle) найдена локально")
        elif entry and entry.parquet_ohlcv and entry.parquet_ohlcv.exists():
            self._bundle_label.setText("Модель: нет · данные есть")
            self._status_label.setStyleSheet("color: #f9a825; font-weight: bold;")
            self._status_label.setToolTip("OHLCV есть, bundle не обучен — вкладка «Задачи»")
        else:
            self._bundle_label.setText("Модель: нет")
            self._status_label.setStyleSheet("color: #c62828; font-weight: bold;")
            self._status_label.setToolTip("Нет локальных данных для этой пары/ТФ")

    def current_symbol(self) -> str:
        data = self._symbol_combo.currentData()
        if data:
            return str(data)
        text = self._symbol_combo.currentText().strip()
        return text or "BTC/USDT"

    def current_timeframe(self) -> str:
        return self._tf_combo.currentText() or "1h"

    def _emit_context(self) -> None:
        self.symbol_changed.emit(self.current_symbol(), self.current_timeframe())
