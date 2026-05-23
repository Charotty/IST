"""Run details: summary HTML + raw JSON."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from PyQt6.QtWidgets import QPlainTextEdit, QTabWidget, QTextBrowser, QVBoxLayout, QWidget

from gui.app.backtest_run_format import format_run_summary_html


class RunSnapshotPanel(QWidget):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._tabs = QTabWidget()
        self._summary = QTextBrowser()
        self._summary.setOpenExternalLinks(False)
        self._summary.setPlaceholderText("Выберите прогон в таблице…")

        self._json = QPlainTextEdit()
        self._json.setReadOnly(True)
        self._json.setPlaceholderText("Полный JSON снимка runs/<run_id>.json")

        self._tabs.addTab(self._summary, "Сводка")
        self._tabs.addTab(self._json, "JSON")
        layout.addWidget(self._tabs)

    def clear(self) -> None:
        self._summary.clear()
        self._json.clear()

    def set_error(self, message: str) -> None:
        self.clear()
        self._summary.setPlainText(message)

    def set_run(self, data: Dict[str, Any]) -> None:
        self._summary.setHtml(format_run_summary_html(data))
        self._json.setPlainText(
            json.dumps(data, indent=2, default=str, ensure_ascii=False)
        )
        self._tabs.setCurrentIndex(0)
