"""Pipeline status and CLI job launcher (orchestration)."""

from __future__ import annotations

from typing import List, Optional

from PyQt6.QtCore import QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from gui.api import IstGuiClient
from gui.api.types import DataHealth, PipelineStepStatus
from gui.app.workers import CliProcessWorker, PipelineStatusWorker


class JobsView(QWidget):
    pipeline_finished = pyqtSignal()

    def __init__(self, api: Optional[IstGuiClient] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._symbol = "BTC/USDT"
        self._timeframe = "1h"
        self._api = api or IstGuiClient()
        self._worker: Optional[PipelineStatusWorker] = None
        self._cli: Optional[CliProcessWorker] = None
        self._cli_running = False

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Чеклист pipeline"))
        self._steps = QListWidget()
        self._steps.setMaximumHeight(120)
        layout.addWidget(self._steps)

        self._health = QLabel("")
        self._health.setWordWrap(True)
        layout.addWidget(self._health)

        cli_box = QGroupBox("Запуск CLI — stdout subprocess (orchestration)")
        cli_layout = QVBoxLayout(cli_box)

        row1 = QHBoxLayout()
        self._btn_prepare = QPushButton("Подготовить символ")
        self._btn_features = QPushButton("Признаки")
        self._btn_tune = QPushButton("Тюнинг thesis")
        self._btn_train = QPushButton("Финальное обучение")
        self._btn_report = QPushButton("Отчёт WFO")
        self._btn_stop = QPushButton("Стоп")
        self._btn_stop.setEnabled(False)
        row1.addWidget(self._btn_prepare)
        row1.addWidget(self._btn_features)
        row1.addWidget(self._btn_tune)
        row1.addWidget(self._btn_train)
        row1.addWidget(self._btn_report)
        row1.addWidget(self._btn_stop)
        cli_layout.addLayout(row1)

        row1b = QHBoxLayout()
        row1b.addWidget(QLabel("Фаза tune-thesis:"))
        self._tune_phase = QComboBox()
        self._tune_phase.addItems(["all", "fast", "refine", "confirm"])
        row1b.addWidget(self._tune_phase)
        row1b.addStretch()
        cli_layout.addLayout(row1b)

        self._tune_config_hint = QLabel(
            "tune-thesis → --config config/profiles/canonical_4model.yaml "
            "(уровни: config/profiles/thesis_tuning.yaml)"
        )
        self._tune_config_hint.setWordWrap(True)
        self._tune_config_hint.setStyleSheet("color: #666; font-size: 11px;")
        cli_layout.addWidget(self._tune_config_hint)

        row2 = QHBoxLayout()
        self._download = QCheckBox("Перекачать OHLCV")
        row2.addWidget(self._download)
        row2.addWidget(QLabel("Баров (0=YAML):"))
        self._max_rows = QSpinBox()
        self._max_rows.setRange(0, 50000)
        self._max_rows.setValue(0)
        self._max_rows.setSpecialValueText("YAML")
        self._max_rows.setSingleStep(500)
        row2.addWidget(self._max_rows)
        self._use_cache = QCheckBox("Кэш фичей")
        self._use_cache.setChecked(True)
        row2.addWidget(self._use_cache)
        self._use_tuning = QCheckBox("tuning-best")
        self._use_tuning.setChecked(True)
        row2.addWidget(self._use_tuning)
        self._full_models = QCheckBox("Все 4 модели (TF)")
        row2.addWidget(self._full_models)
        row2.addStretch()
        cli_layout.addLayout(row2)

        self._report_hint = QLabel(
            "Отчёт WFO: при включённом tuning-best — эталон thesis_4model + overrides; "
            "без флага — canonical (не для дипломного отчёта)."
        )
        self._report_hint.setWordWrap(True)
        self._report_hint.setStyleSheet("color: #666; font-size: 11px;")
        cli_layout.addWidget(self._report_hint)

        self._cli_status = QLabel("")
        self._cli_status.setStyleSheet("color: #1565c0; font-weight: bold;")
        cli_layout.addWidget(self._cli_status)

        self._cli_log = QPlainTextEdit()
        self._cli_log.setReadOnly(True)
        self._cli_log.setPlaceholderText("Вывод python -m orchestration …")
        cli_layout.addWidget(self._cli_log, stretch=1)
        layout.addWidget(cli_box, stretch=1)

        self._btn_prepare.clicked.connect(self._run_prepare)
        self._btn_features.clicked.connect(self._run_build_features)
        self._btn_tune.clicked.connect(self._run_tune_thesis)
        self._btn_train.clicked.connect(self._run_train_final)
        self._btn_report.clicked.connect(self._run_report_real)
        self._btn_stop.clicked.connect(self._stop_cli)

        self._sync_cli_buttons()

    def _sync_cli_buttons(self) -> None:
        demo = getattr(self._api, "demo", False)
        can_start = (not self._cli_running) and (not demo)
        cli_btns = (
            self._btn_prepare,
            self._btn_features,
            self._btn_tune,
            self._btn_train,
            self._btn_report,
        )
        for b in cli_btns:
            b.setEnabled(can_start)
        self._btn_stop.setEnabled(self._cli_running)
        if demo:
            self._cli_log.setPlaceholderText(
                "В демо-режиме CLI отключён. Используйте реальный запуск без --demo."
            )

    def set_context(self, symbol: str, timeframe: str) -> None:
        self._symbol = symbol
        self._timeframe = timeframe

    def refresh(self) -> None:
        if self._worker and self._worker.isRunning():
            return
        self._steps.clear()
        self._worker = PipelineStatusWorker(self._symbol, self._timeframe, api=self._api)
        self._worker.finished.connect(self._on_status)
        self._worker.failed.connect(lambda m: self._health.setText(f"Ошибка: {m}"))
        self._worker.start()

    def _on_status(self, payload: list) -> None:
        steps: List[PipelineStepStatus] = payload[0]
        health: DataHealth = payload[1]
        self._steps.clear()
        for st in steps:
            mark = "✓" if st.done else "✗"
            self._steps.addItem(f"{mark}  {st.name}: {st.detail}")
        self._health.setText(
            f"OHLCV: {health.ohlcv_rows} строк, последний {health.ohlcv_last_ts}\n"
            f"Признаки: {health.features_rows} строк, последний {health.features_last_ts}"
        )

    def _detach_cli_worker(self) -> None:
        if self._cli is None:
            return
        try:
            self._cli.line.disconnect(self._cli_log.appendPlainText)
        except TypeError:
            pass
        try:
            self._cli.finished_code.disconnect()
        except TypeError:
            pass
        try:
            self._cli.finished.disconnect()
        except TypeError:
            pass

    def _finish_cli(self, code: int, title: str) -> None:
        if not self._cli_running:
            return
        self._cli_running = False
        self._sync_cli_buttons()
        self._cli_log.appendPlainText(f"\n--- {title}: завершено (код {code}) ---")
        self._cli_status.setText("" if code == 0 else f"Завершено с кодом {code}")
        self._detach_cli_worker()
        self._cli = None
        QTimer.singleShot(0, self.refresh)
        self.pipeline_finished.emit()

    def _start_cli(self, argv: List[str], title: str) -> None:
        if getattr(self._api, "demo", False):
            self._cli_log.appendPlainText("CLI недоступен в демо-режиме.")
            return
        if self._cli_running:
            self._cli_log.appendPlainText("Уже выполняется другая команда.")
            return

        self._detach_cli_worker()
        self._cli_log.clear()
        mode = ""
        if title == "report-real":
            mode = (
                "режим=эталон (tuning-best)"
                if self._use_tuning.isChecked()
                else "режим=canonical (без tuning-best)"
            )
        self._cli_log.appendPlainText(f"$ {' '.join(argv)}\n")
        if mode:
            self._cli_log.appendPlainText(f"# {mode}\n")
        self._cli_running = True
        self._sync_cli_buttons()
        self._cli_status.setText(f"Выполняется: {title}…")

        self._cli = CliProcessWorker(argv, cwd=str(self._api.cli.repo_cwd()))
        self._cli.line.connect(self._cli_log.appendPlainText)
        self._cli.finished_code.connect(
            lambda c, t=title: self._finish_cli(c, t)
        )
        self._cli.finished.connect(self._on_cli_thread_exited)
        self._cli.start()

    def _on_cli_thread_exited(self) -> None:
        """Подстраховка: поток завершился, а finished_code не пришёл."""
        if self._cli_running:
            self._finish_cli(-1, "cli")

    def _stop_cli(self) -> None:
        if self._cli and self._cli.isRunning():
            self._cli.stop_process()
            self._cli_log.appendPlainText("Остановка…")
        if self._cli_running:
            self._cli_status.setText("Остановлено пользователем")
            self._finish_cli(-1, "остановлено")

    def _run_prepare(self) -> None:
        argv = self._api.cli.prepare_symbol_cmd(
            self._symbol,
            self._timeframe,
            download=self._download.isChecked(),
        )
        self._start_cli(argv, "prepare-symbol")

    def _run_train_final(self) -> None:
        argv = self._api.cli.train_final_cmd(self._symbol, self._timeframe, force=True)
        self._start_cli(argv, "train-final-symbol")

    def _run_build_features(self) -> None:
        argv = self._api.cli.build_features_cmd(self._symbol, self._timeframe)
        self._start_cli(argv, "build-features")

    def _run_tune_thesis(self) -> None:
        argv = self._api.cli.tune_thesis_cmd(
            self._symbol,
            self._timeframe,
            phase=self._tune_phase.currentText(),
        )
        self._start_cli(argv, "tune-thesis")

    def _run_report_real(self) -> None:
        argv = self._api.cli.report_real_cmd(
            self._symbol,
            self._timeframe,
            max_rows=self._max_rows.value(),
            full_models=self._full_models.isChecked(),
            use_tuning_best=self._use_tuning.isChecked(),
            use_feature_cache=self._use_cache.isChecked(),
        )
        self._start_cli(argv, "report-real")
