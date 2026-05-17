"""Entry: ``python -m gui.app`` or ``python -m gui.app.main``."""

from __future__ import annotations

import sys


def _settings_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).lower() in ("1", "true", "yes", "on")


def demo_enabled_from_env() -> bool:
    import os

    if "--demo" in sys.argv:
        return True
    if os.environ.get("IST_GUI_DEMO", "").strip() in ("1", "true", "yes"):
        return True
    try:
        from PyQt6.QtCore import QSettings

        return _settings_bool(QSettings("IST", "Desktop").value("demo_mode"))
    except Exception:
        return False


def main() -> int:
    try:
        from PyQt6.QtWidgets import QApplication
    except ImportError:
        print("PyQt6 required: pip install -r requirements-gui.txt", file=sys.stderr)
        return 1

    demo = demo_enabled_from_env()
    app = QApplication(sys.argv)
    app.setApplicationName("IST")
    app.setOrganizationName("IST")

    from gui.api import IstGuiClient
    from gui.app.main_window import MainWindow

    window = MainWindow(IstGuiClient(demo=demo))
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
