"""Unit tests for orchestration/feature_store.py."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from orchestration.feature_store import (
    _columns_hash,
    _hash_sections,
    is_stale,
    manifest_path,
    write_manifest,
)
from orchestration.canonical_pipeline import resolve_profile_path


def test_hash_sections_stable():
    profile = resolve_profile_path("config/profiles/canonical_4model.yaml")
    h1 = _hash_sections(profile)
    h2 = _hash_sections(profile)
    assert h1 == h2
    assert len(h1) == 16


def test_columns_hash_changes_with_columns():
    df1 = pd.DataFrame({"a": [1], "b": [2]})
    df2 = pd.DataFrame({"a": [1], "c": [2]})
    assert _columns_hash(df1) != _columns_hash(df2)


def test_is_stale_missing_parquet(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "orchestration.feature_store.features_parquet_for",
        lambda sym, tf: tmp_path / "missing.parquet",
    )
    assert is_stale("BTC/USDT", "1h") is True


def test_write_and_read_manifest(tmp_path, monkeypatch):
    feat = tmp_path / "BTC-USDT_1h.parquet"
    df = pd.DataFrame({"x": [1.0, 2.0], "close": [1.0, 2.0]})
    df.to_parquet(feat)

    def _feat_path(sym, tf):
        return feat

    def _man_path(sym, tf):
        return tmp_path / "BTC-USDT_1h.manifest.json"

    monkeypatch.setattr("orchestration.feature_store.features_parquet_for", _feat_path)
    monkeypatch.setattr("orchestration.feature_store.manifest_path", _man_path)

    meta = write_manifest("BTC/USDT", "1h", df, config_path="config/profiles/canonical_4model.yaml")
    assert meta["row_count"] == 2
    loaded = json.loads(_man_path("BTC/USDT", "1h").read_text(encoding="utf-8"))
    assert loaded["columns_hash"] == meta["columns_hash"]
