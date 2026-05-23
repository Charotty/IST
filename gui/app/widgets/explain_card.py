"""Overview card: last-bar decision from ExplainSnapshot."""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from gui.api.types import ExplainSnapshot


class ExplainCard(QGroupBox):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__("Решение (последний бар)", parent)
        root = QVBoxLayout(self)

        top = QHBoxLayout()
        self._direction = QLabel("—")
        self._direction.setStyleSheet("font-size: 22px; font-weight: bold;")
        top.addWidget(self._direction)
        top.addStretch()
        self._regime = QLabel("режим: —")
        top.addWidget(self._regime)
        root.addLayout(top)

        form = QFormLayout()
        self._as_of = QLabel("—")
        self._close = QLabel("—")
        self._meta = QLabel("—")
        self._confidence = QLabel("—")
        self._confidence.setToolTip(
            "Уверенность = |P(up) − 0.5| × 2 (шкала 0…1).\n"
            "0.23 ≈ 23% «силы» сигнала от нейтрали 50/50, не 0.23%."
        )
        self._signal = QLabel("—")
        self._pos = QLabel("—")
        self._blocked = QLabel("")
        self._blocked.setWordWrap(True)
        self._blocked.setStyleSheet("color: #c62828;")
        form.addRow("Время:", self._as_of)
        form.addRow("Close:", self._close)
        form.addRow("Meta P(up):", self._meta)
        form.addRow("Уверенность:", self._confidence)
        form.addRow("Сигнал:", self._signal)
        form.addRow("Доля позиции:", self._pos)
        form.addRow("Блокировка:", self._blocked)
        root.addLayout(form)

        self._models = QTableWidget(0, 2)
        self._models.setHorizontalHeaderLabels(["Модель", "P(up)"])
        self._models.horizontalHeader().setStretchLastSection(True)
        self._models.setMaximumHeight(140)
        root.addWidget(QLabel("Прогнозы моделей"))
        root.addWidget(self._models)

        self._weights = QTableWidget(0, 2)
        self._weights.setHorizontalHeaderLabels(["Модель", "Вес"])
        self._weights.horizontalHeader().setStretchLastSection(True)
        self._weights.setMaximumHeight(140)
        root.addWidget(QLabel("Активные веса"))
        root.addWidget(self._weights)

        self._bundle = QLabel("")
        self._bundle.setStyleSheet("color: #555; font-size: 11px;")
        self._bundle.setWordWrap(True)
        root.addWidget(self._bundle)

    def set_loading(self) -> None:
        self._direction.setText("Загрузка…")
        self._blocked.setText("")

    def set_error(self, message: str) -> None:
        self._direction.setText("Ошибка")
        self._blocked.setText(message)

    def set_snapshot(self, snap: ExplainSnapshot) -> None:
        color = {"long": "#2e7d32", "short": "#c62828", "flat": "#666"}.get(
            snap.direction, "#333"
        )
        self._direction.setText(snap.direction.upper())
        self._direction.setStyleSheet(f"font-size: 22px; font-weight: bold; color: {color};")
        self._regime.setText(f"режим: {snap.regime}")

        self._as_of.setText(snap.as_of)
        self._close.setText(
            f"{snap.last_close:,.2f}" if snap.last_close is not None else "—"
        )
        self._meta.setText(f"{snap.meta_probability:.4f}")
        self._confidence.setText(f"{snap.confidence:.4f}")
        self._signal.setText(str(snap.signal))
        self._pos.setText(f"{snap.position_size_frac:.4f}")

        if snap.why_blocked:
            self._blocked.setText(snap.why_blocked)
        else:
            self._blocked.setText("— (сделка разрешена decision-слоем)")

        self._fill_table(self._models, snap.model_probs, fmt_prob=True)
        self._fill_table(self._weights, snap.active_weights, fmt_prob=False)
        self._bundle.setText(f"Бандл: {snap.bundle_dir}")

    @staticmethod
    def _fill_table(table: QTableWidget, data: dict, *, fmt_prob: bool) -> None:
        table.setRowCount(len(data))
        for row, (key, val) in enumerate(sorted(data.items())):
            table.setItem(row, 0, QTableWidgetItem(key))
            text = f"{val:.4f}" if fmt_prob else f"{val:.3f}"
            item = QTableWidgetItem(text)
            item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            table.setItem(row, 1, item)
