"""Smoke tests for paper evidence (no DL if bundle missing)."""

from __future__ import annotations

import pytest

from orchestration.paper_evidence import _why_blocked


def test_why_blocked_dead_zone():
    class C:
        min_signal_margin = 0.06
        direction_threshold = 0.56

    assert _why_blocked(0, 0.5115, C()) == "dead-zone (min_signal_margin)"


def test_why_blocked_none_when_signal():
    class C:
        min_signal_margin = 0.06
        direction_threshold = 0.56

    assert _why_blocked(1, 0.6, C()) is None
