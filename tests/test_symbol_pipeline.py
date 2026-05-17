"""Юнит-тесты пер-символьного контура (без TensorFlow и тяжёлых обучений)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtesting.results_journal import BacktestResultsJournal
from orchestration import symbols as sym_mod
from orchestration.symbols import (
    list_known_symbols,
    merged_config,
    normalize_symbol,
    paths_for,
    slug,
    tuning_best_for,
    write_symbol_config,
)


@pytest.fixture
def isolated_repo(tmp_path, monkeypatch):
    """Изолируем DATA_DIR / SYMBOLS_DIR / ARTIFACTS_DIR в tmp."""
    data = tmp_path / "data" / "ohlcv"
    symbols_dir = tmp_path / "config" / "symbols"
    artifacts = tmp_path / "artifacts"
    base_yaml = tmp_path / "config.yaml"
    for d in (data, symbols_dir, artifacts):
        d.mkdir(parents=True, exist_ok=True)
    base_yaml.write_text(
        "orchestration_tuning_best:\n  min_signal_margin: 0.07\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(sym_mod, "DATA_DIR", data)
    monkeypatch.setattr(sym_mod, "SYMBOLS_DIR", symbols_dir)
    monkeypatch.setattr(sym_mod, "ARTIFACTS_DIR", artifacts)
    monkeypatch.setattr(sym_mod, "BASE_CONFIG", base_yaml)
    return tmp_path


def test_normalize_and_slug():
    assert normalize_symbol("btc/usdt") == "BTC-USDT"
    assert normalize_symbol("ETH_USDT") == "ETH-USDT"
    assert slug("BTC/USDT", "1h") == "BTC-USDT_1h"


def test_paths_for_returns_consistent_paths(isolated_repo):
    sp = paths_for("BTC/USDT", "1h")
    assert sp.slug == "BTC-USDT_1h"
    assert sp.parquet.parent == isolated_repo / "data" / "ohlcv"
    assert sp.config_yaml.parent == isolated_repo / "config" / "symbols"
    assert sp.artifacts_root == isolated_repo / "artifacts" / "BTC-USDT_1h"


def test_write_and_read_per_symbol_config(isolated_repo):
    write_symbol_config(
        "ETH/USDT",
        "1h",
        tuning_best={"min_signal_margin": 0.09, "trade_mode": "long_only"},
    )
    eff = tuning_best_for("ETH/USDT", "1h")
    assert eff["min_signal_margin"] == 0.09
    # base config is fallback if no per-symbol file
    btc = tuning_best_for("BTC/USDT", "1h")
    assert btc["min_signal_margin"] == 0.07


def test_merged_config_overrides(isolated_repo):
    write_symbol_config(
        "SOL/USDT",
        "1h",
        tuning_best={"min_signal_margin": 0.11},
        extra={"backtesting": {"acceptance": {"min_oos_sharpe": 0.7}}},
    )
    cfg = merged_config("SOL/USDT", "1h")
    assert cfg["orchestration_tuning_best"]["min_signal_margin"] == 0.11
    assert cfg["backtesting"]["acceptance"]["min_oos_sharpe"] == 0.7


def test_list_known_symbols(isolated_repo):
    (isolated_repo / "data" / "ohlcv" / "BTC-USDT_1h.parquet").write_bytes(b"x")
    write_symbol_config("ETH/USDT", "1h")
    rows = list_known_symbols()
    slugs = {r["slug"] for r in rows}
    assert {"BTC-USDT_1h", "ETH-USDT_1h"} <= slugs


def test_journal_filters_by_symbol(tmp_path):
    j = BacktestResultsJournal(tmp_path / "j")
    base_report = {
        "fold_metrics": [],
        "summary": {"n_folds": 1, "mean_sharpe": 0.5},
        "criteria": {
            "acceptance": {"passed": True, "checks": []},
            "target": {"passed": False, "checks": []},
        },
        "parquet": "x",
        "feature_rows": 100,
        "max_rows": 100,
    }
    j.append_run(
        {**base_report, "symbol": "BTC-USDT", "timeframe": "1h", "stage": "holdout"},
        label="btc-h",
    )
    j.append_run(
        {**base_report, "symbol": "ETH-USDT", "timeframe": "1h", "stage": "holdout"},
        label="eth-h",
    )
    j.append_run(
        {**base_report, "symbol": "BTC-USDT", "timeframe": "1h", "stage": "tune"},
        label="btc-t",
    )

    btc_holdout = j.list_runs(symbol="BTC-USDT", timeframe="1h", stage="holdout")
    assert len(btc_holdout) == 1
    assert btc_holdout[0]["label"] == "btc-h"

    best_eth = j.best_acceptance_run(symbol="ETH-USDT")
    assert best_eth and best_eth["label"] == "eth-h"


def test_split_train_holdout_chronological():
    from orchestration.symbol_pipeline import split_train_holdout

    n = 1000
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    df = pd.DataFrame({"close": np.linspace(100, 200, n)}, index=idx)
    train, hold = split_train_holdout(df, holdout_fraction=0.2)
    assert len(train) == 800
    assert len(hold) == 200
    assert train.index[-1] < hold.index[0]


def test_symbol_manifest_serialization(tmp_path, isolated_repo):
    from orchestration.symbol_pipeline import (
        SymbolManifest,
        read_symbol_manifest,
        write_symbol_manifest,
    )

    sp = paths_for("BTC/USDT", "1h")
    m = SymbolManifest(
        symbol=sp.symbol,
        timeframe=sp.timeframe,
        slug=sp.slug,
        created_at="2026-05-16T00:00:00+00:00",
        parquet=str(sp.parquet),
        train_span={"start": "2022-01-01", "end": "2025-01-01", "rows": 26000},
        holdout_span={"start": "2025-01-01", "end": "2026-01-01", "rows": 8760},
        full_span={"start": "2022-01-01", "end": "2026-01-01", "rows": 34760},
        tuning_best_params={"min_signal_margin": 0.08},
        tune_run_ids=["a", "b"],
        holdout_run_id="c",
        holdout_summary={"mean_sharpe": 0.55},
        holdout_acceptance_passed=True,
        holdout_target_passed=False,
        artifact_bundle=str(sp.artifacts_root / "20260516_run01"),
        bundle_run_id="20260516_run01",
        ready_for_paper=True,
        ready_for_live=False,
    )
    out = write_symbol_manifest(sp, m)
    assert out.is_file()
    raw = json.loads(out.read_text(encoding="utf-8"))
    assert raw["ready_for_paper"] is True
    assert raw["holdout_acceptance_passed"] is True
    again = read_symbol_manifest(sp)
    assert again["bundle_run_id"] == "20260516_run01"


def test_latest_pointer(tmp_path, isolated_repo):
    sp = paths_for("BTC/USDT", "1h")
    sp.artifacts_root.mkdir(parents=True, exist_ok=True)
    (sp.artifacts_root / "20260516T120000Z_aaa").mkdir()
    (sp.artifacts_root / "20260516T130000Z_bbb").mkdir()
    sp.write_latest_pointer("20260516T130000Z_bbb")
    assert sp.latest_bundle().name == "20260516T130000Z_bbb"
