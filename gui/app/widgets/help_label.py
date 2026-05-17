"""Поясняющие блоки с читаемым чёрным текстом (в т.ч. rich HTML)."""

from __future__ import annotations

from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QLabel


def help_label(html: str, *, background: str, border: str, padding: str = "10px") -> QLabel:
    lbl = QLabel(f'<div style="color:#000000;">{html}</div>')
    lbl.setWordWrap(True)
    lbl.setStyleSheet(
        f"color: #000000; background: {background}; padding: {padding}; "
        f"border-radius: 6px; border: 1px solid {border};"
    )
    pal = lbl.palette()
    pal.setColor(QPalette.ColorRole.WindowText, QColor(0, 0, 0))
    pal.setColor(QPalette.ColorRole.Text, QColor(0, 0, 0))
    lbl.setPalette(pal)
    return lbl
