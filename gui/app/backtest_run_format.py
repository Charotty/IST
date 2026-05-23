"""Format backtest journal run snapshots for the GUI."""

from __future__ import annotations

import html
from typing import Any, Dict, List, Optional

from gui.app.formatters import format_pass_fail


def _f(val: Any, *, digits: int = 3, suffix: str = "") -> str:
    if val is None:
        return "—"
    try:
        return f"{float(val):.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return "—"


def _checks_html(checks: Optional[List[dict]], title: str) -> str:
    if not checks:
        return f"<p><i>{html.escape(title)}: нет данных</i></p>"
    lines = [f"<p><b>{html.escape(title)}</b></p><ul>"]
    for c in checks:
        if not isinstance(c, dict):
            continue
        ok = bool(c.get("passed"))
        color = "#2e7d32" if ok else "#c62828"
        mark = "✓" if ok else "✗"
        name = html.escape(str(c.get("name", "?")))
        detail = html.escape(str(c.get("detail", "")))
        lines.append(
            f'<li style="color:{color}">{mark} <b>{name}</b> — {detail}</li>'
        )
    lines.append("</ul>")
    return "".join(lines)


def format_run_summary_html(data: Dict[str, Any]) -> str:
    """Human-readable summary for ``RunSnapshotPanel``."""
    s = data.get("summary") or {}
    params = data.get("params") or {}
    acc = data.get("acceptance_passed")
    tgt = data.get("target_passed")
    models = params.get("model_keys") or []

    head = (
        f"<h3 style='margin:0 0 8px 0'>{html.escape(str(data.get('run_id', '')))}</h3>"
        f"<p style='margin:0 0 10px 0;color:#555'>"
        f"{html.escape(str(data.get('symbol', '?')))} "
        f"{html.escape(str(data.get('timeframe', '')))} · "
        f"{html.escape(str(data.get('timestamp_utc', ''))[:19])} · "
        f"label: <code>{html.escape(str(data.get('label') or data.get('stage') or '—'))}</code>"
        f"</p>"
        f"<p><b>Acceptance:</b> {format_pass_fail(acc)} &nbsp; "
        f"<b>Target:</b> {format_pass_fail(tgt)}</p>"
    )

    metrics = (
        "<table cellspacing='4' cellpadding='2'>"
        f"<tr><td>Фолдов OOS</td><td><b>{s.get('n_folds', '—')}</b></td></tr>"
        f"<tr><td>Mean Sharpe</td><td><b>{_f(s.get('mean_sharpe'))}</b></td></tr>"
        f"<tr><td>Profit factor</td><td><b>{_f(s.get('mean_profit_factor'))}</b></td></tr>"
        f"<tr><td>WFE</td><td><b>{_f(s.get('mean_wfe'))}</b></td></tr>"
        f"<tr><td>Return OOS %</td><td><b>{_f(s.get('mean_total_return_pct'))}%</b></td></tr>"
        f"<tr><td>Worst DD %</td><td><b>{_f(s.get('worst_max_drawdown_pct'))}%</b></td></tr>"
        f"<tr><td>Recovery</td><td><b>{_f(s.get('mean_recovery_factor'))}</b></td></tr>"
        f"<tr><td>Сделок (сумма)</td><td><b>{s.get('total_trade_events', '—')}</b></td></tr>"
        f"<tr><td>Фолдов с return&gt;0</td><td><b>{s.get('folds_positive_return', '—')}</b></td></tr>"
        "</table>"
    )

    prof = (
        f"<p style='margin-top:10px'><b>Профиль:</b> "
        f"<code>{html.escape(str(data.get('profile') or params.get('profile', '—')))}</code><br>"
        f"<b>Модели:</b> {html.escape(', '.join(models) if models else '—')}<br>"
        f"<b>Строк фичей:</b> {data.get('feature_rows', '—')} · "
        f"<b>max_rows:</b> {data.get('max_rows', '—')}</p>"
    )

    return (
        head
        + metrics
        + prof
        + _checks_html(data.get("acceptance_checks"), "Критерии acceptance")
        + _checks_html(data.get("target_checks"), "Критерии target")
    )


def stage_display(run_label: Optional[str], run_stage: Optional[str]) -> str:
    return (run_stage or run_label or "—").strip() or "—"
