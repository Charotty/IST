"""Конфигурация — редактирование и проверка per-symbol YAML."""

from __future__ import annotations

import json
from typing import Optional

from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from gui.api import IstGuiClient
from gui.app.widgets.help_label import help_label
from gui.app.workers import ExplainWorker


class SettingsView(QWidget):
    def __init__(self, api: Optional[IstGuiClient] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._api = api or IstGuiClient()
        self._symbol = "BTC/USDT"
        self._timeframe = "1h"
        self._settings = QSettings("IST", "Desktop")
        self._explain_worker: Optional[ExplainWorker] = None

        outer = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        layout = QVBoxLayout(body)

        head = help_label(
            "Параметры сохраняются в <code>config/symbols/&lt;пара&gt;_&lt;тф&gt;.yaml</code> "
            "(секция <code>orchestration_tuning_best</code>). "
            "После сохранения можно сразу проверить inference.",
            background="#fafafa",
            border="#e0e0e0",
            padding="8px",
        )
        layout.addWidget(head)

        self._demo_cb = QCheckBox("Синтетические данные ML (скрытый demo, перезапуск)")
        self._demo_cb.setChecked(bool(self._api.demo))
        self._demo_cb.setToolTip("py -3 -m gui.app --demo или переменная IST_GUI_DEMO=1")
        self._demo_cb.toggled.connect(self._on_demo_toggled)
        layout.addWidget(self._demo_cb)

        form_box = QGroupBox("Decision и ансамбль")
        form = QFormLayout(form_box)

        self._direction_thr = QDoubleSpinBox()
        self._direction_thr.setRange(0.5, 0.65)
        self._direction_thr.setDecimals(3)
        self._direction_thr.setSingleStep(0.01)
        form.addRow("Порог направления:", self._direction_thr)

        self._margin = QDoubleSpinBox()
        self._margin.setRange(0.0, 0.2)
        self._margin.setDecimals(3)
        self._margin.setSingleStep(0.005)
        form.addRow("Мёртвая зона (margin):", self._margin)

        self._trade_mode = QComboBox()
        self._trade_mode.addItems(["both", "long_only", "short_only"])
        form.addRow("Режим сделок:", self._trade_mode)

        self._max_pos = QDoubleSpinBox()
        self._max_pos.setRange(0.05, 1.0)
        self._max_pos.setDecimals(2)
        self._max_pos.setSingleStep(0.05)
        form.addRow("Макс. доля позиции:", self._max_pos)

        self._ensemble = QComboBox()
        self._ensemble.addItems(["regime_adaptive", "regime_weighted", "fixed"])
        form.addRow("Режим ансамбля:", self._ensemble)

        layout.addWidget(form_box)

        btn_row = QHBoxLayout()
        self._btn_load = QPushButton("Загрузить")
        self._btn_save = QPushButton("Сохранить")
        self._btn_save.setStyleSheet("font-weight: bold;")
        self._btn_test = QPushButton("Тест: inference")
        self._btn_check = QPushButton("Проверить pipeline")
        btn_row.addWidget(self._btn_load)
        btn_row.addWidget(self._btn_save)
        btn_row.addWidget(self._btn_test)
        btn_row.addWidget(self._btn_check)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self._result = QPlainTextEdit()
        self._result.setReadOnly(True)
        self._result.setMaximumHeight(140)
        self._result.setPlaceholderText("Результат теста появится здесь…")
        layout.addWidget(self._result)

        tabs = QTabWidget()
        self._yaml_edit = QPlainTextEdit()
        self._yaml_edit.setPlaceholderText("Полный YAML override символа…")
        tabs.addTab(self._yaml_edit, "YAML (продвинутый)")

        self._merged_view = QPlainTextEdit()
        self._merged_view.setReadOnly(True)
        tabs.addTab(self._merged_view, "Итоговый merged config")

        layout.addWidget(tabs, stretch=1)

        scroll.setWidget(body)
        outer.addWidget(scroll)

        self._btn_load.clicked.connect(self._load_form)
        self._btn_save.clicked.connect(self._save)
        self._btn_test.clicked.connect(self._test_inference)
        self._btn_check.clicked.connect(self._check_pipeline)

    def _on_demo_toggled(self, checked: bool) -> None:
        self._settings.setValue("demo_mode", checked)
        self._settings.sync()

    def set_context(self, symbol: str, timeframe: str) -> None:
        self._symbol = symbol
        self._timeframe = timeframe

    def refresh(self) -> None:
        self._demo_cb.blockSignals(True)
        self._demo_cb.setChecked(bool(self._api.demo))
        self._demo_cb.blockSignals(False)
        self._load_form()

    def _load_form(self) -> None:
        try:
            orch = self._api.config.orchestration_section(self._symbol, self._timeframe)
            self._direction_thr.setValue(float(orch.get("direction_threshold", 0.52)))
            self._margin.setValue(float(orch.get("min_signal_margin", 0.0)))
            mode = str(orch.get("trade_mode", "both"))
            idx = self._trade_mode.findText(mode)
            if idx >= 0:
                self._trade_mode.setCurrentIndex(idx)
            self._max_pos.setValue(float(orch.get("max_position_fraction", 1.0)))
            ens = str(orch.get("ensemble_mode", "regime_adaptive"))
            idx = self._ensemble.findText(ens)
            if idx >= 0:
                self._ensemble.setCurrentIndex(idx)

            text, _ = self._api.config.load_raw_yaml(self._symbol, self._timeframe)
            self._yaml_edit.setPlainText(text)
            merged = self._api.config.merged(self._symbol, self._timeframe)
            self._merged_view.setPlainText(json.dumps(merged, indent=2, ensure_ascii=False, default=str))
            self._result.setPlainText(f"Загружено для {self._symbol} {self._timeframe}")
        except Exception as e:
            self._result.setPlainText(f"Ошибка загрузки: {e}")

    def _collect_values(self) -> dict:
        return {
            "direction_threshold": self._direction_thr.value(),
            "min_signal_margin": self._margin.value(),
            "trade_mode": self._trade_mode.currentText(),
            "max_position_fraction": self._max_pos.value(),
            "ensemble_mode": self._ensemble.currentText(),
            "model_keys": ["lgb", "gru", "xgb", "cnn"],
            "apply_decision_pipeline": True,
        }

    def _save(self) -> None:
        try:
            path = self._api.config.save_orchestration(
                self._symbol, self._timeframe, self._collect_values()
            )
            yaml_text = self._yaml_edit.toPlainText().strip()
            if yaml_text:
                path = self._api.config.save_raw_yaml(self._symbol, self._timeframe, yaml_text)
            self._result.setPlainText(f"Сохранено: {path}")
            QMessageBox.information(self, "Конфигурация", f"Сохранено:\n{path}")
            self._load_form()
        except Exception as e:
            QMessageBox.warning(self, "Ошибка", str(e))

    def _check_pipeline(self) -> None:
        try:
            r = self._api.config.test_pipeline_ready(self._symbol, self._timeframe)
            lines = [f"{k}: {'OK' if v else 'нет'}" for k, v in r["checks"].items()]
            lines.append(f"bundle: {r.get('bundle_run_id') or '—'}")
            self._result.setPlainText("\n".join(lines))
        except Exception as e:
            self._result.setPlainText(str(e))

    def _test_inference(self) -> None:
        if self._explain_worker and self._explain_worker.isRunning():
            return
        self._result.setPlainText("Запуск inference…")
        self._explain_worker = ExplainWorker(
            self._symbol, self._timeframe, window=256, api=self._api
        )
        self._explain_worker.finished.connect(self._on_test_ok)
        self._explain_worker.failed.connect(
            lambda m: self._result.setPlainText(f"Inference ошибка:\n{m}")
        )
        self._explain_worker.start()

    def _on_test_ok(self, snap) -> None:
        self._result.setPlainText(
            f"OK — {snap.direction} signal={snap.signal} "
            f"meta={snap.meta_probability:.4f} regime={snap.regime}\n"
            f"blocked: {snap.why_blocked or 'нет'}"
        )
