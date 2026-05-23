"""Save GUI tab screenshots for diploma section 3.11 (requires display / Windows session)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "figures" / "3_11" / "gui"
os.environ.setdefault("PYTHONPATH", str(ROOT))

# Avoid blocking on OKX network during market load
os.environ.setdefault("IST_GUI_DEMO", "1")


def main() -> int:
    try:
        from PyQt6.QtCore import QTimer
        from PyQt6.QtWidgets import QApplication
    except ImportError:
        print("PyQt6 not installed; pip install -r requirements-gui.txt")
        return 1

    from gui.api import IstGuiClient
    from gui.app.main_window import MainWindow
    from gui.app import i18n_ru as ru

    OUT.mkdir(parents=True, exist_ok=True)
    app = QApplication(sys.argv)
    win = MainWindow(IstGuiClient(demo=True))
    win.show()

    captures = [
        (ru.TAB_CHART, "gui_chart_signals.png"),
        (ru.TAB_JOBS, "gui_jobs_pipeline.png"),
        (ru.TAB_CONFIG, "gui_config.png"),
    ]

    idx = {"i": 0}

    def _shot() -> None:
        i = idx["i"]
        if i >= len(captures):
            QTimer.singleShot(300, app.quit)
            return
        tab_name, fname = captures[i]
        tabs = win._tabs
        for t in range(tabs.count()):
            if tabs.tabText(t) == tab_name:
                tabs.setCurrentIndex(t)
                break
        win._refresh_current_tab()

        def _grab() -> None:
            path = OUT / fname
            win.grab().save(str(path))
            print("Saved", path)
            idx["i"] += 1
            QTimer.singleShot(800, _shot)

        QTimer.singleShot(1500, _grab)

    QTimer.singleShot(2000, _shot)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
