"""Display helpers for GUI tables and labels."""

from __future__ import annotations

from typing import Any, Optional


def format_pass_fail(value: Optional[bool]) -> str:
    if value is None:
        return "—"
    return "PASS" if value else "FAIL"


def format_mean_sharpe(summary: Optional[dict]) -> str:
    if not summary:
        return "—"
    raw = summary.get("mean_sharpe")
    if raw is None:
        return "—"
    try:
        return f"{float(raw):.3f}"
    except (TypeError, ValueError):
        return "—"
