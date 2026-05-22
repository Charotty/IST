"""Unit tests for multilevel thesis tuning helpers."""

from __future__ import annotations

from orchestration.thesis_tuning import (
    FoldPruner,
    iter_level_candidates,
    params_for_level,
    save_shortlist,
    should_prune_trial,
)


def test_params_for_level_fast_has_tune_level():
    p = params_for_level("fast")
    assert p["tune_level"] == "fast"
    assert p.get("max_wfo_folds") == 8


def test_params_for_level_confirm_quality():
    p = params_for_level("confirm")
    assert p["tune_level"] == "confirm"
    assert p.get("max_wfo_folds") == 0


def test_merge_candidate_confirm_keeps_full_wfo():
    from orchestration.thesis_tuning import merge_candidate_params, params_for_level

    base = params_for_level("confirm")
    shortlist = {
        "max_wfo_folds": 8,
        "dl_epochs": 2,
        "max_rows": 6000,
        "walk_forward_step": 400,
        "min_signal_margin": 0.06,
        "ensemble_mode": "regime_adaptive",
    }
    merged = merge_candidate_params("confirm", base, shortlist)
    assert merged["max_wfo_folds"] == 0
    assert merged["dl_epochs"] == 3
    assert merged["max_rows"] == 8000
    assert merged["walk_forward_step"] == 360
    assert merged["min_signal_margin"] == 0.06
    assert merged["ensemble_mode"] == "regime_adaptive"


def test_merge_candidate_patch_keeps_use_risk_bridge():
    from orchestration.thesis_tuning import merge_candidate_params, params_for_level

    base = params_for_level("confirm")
    merged = merge_candidate_params("confirm", base, {"ensemble_mode": "fixed_range"})
    assert merged["use_risk_bridge"] is False
    assert merged["regime"] == "momentum"
    assert merged["max_wfo_folds"] == 0


def test_iter_fast_candidates_count():
    cands = iter_level_candidates("fast", n_trials=5)
    assert len(cands) == 5
    assert all(c.get("tune_level") == "fast" for c in cands)


def test_fold_pruner_below_median():
    pruner = FoldPruner(min_folds=2, min_completed_trials=2)
    pruner.record_completed_trials([0.5, 0.6, 0.55])
    assert pruner.should_prune([{"Sharpe Ratio": -1.0}, {"Sharpe Ratio": -0.5}]) is True
    assert pruner.should_prune([{"Sharpe Ratio": 0.8}, {"Sharpe Ratio": 0.9}]) is False


def test_fold_pruner_warmup_not_before_min_completed():
    pruner = FoldPruner(min_folds=2, min_completed_trials=5)
    pruner.record_completed_trials([1.5])
    assert pruner.should_prune([{"Sharpe Ratio": -1.0}, {"Sharpe Ratio": -0.5}]) is False


def test_should_prune_trial_disabled():
    assert (
        should_prune_trial(
            [{"sharpe_ratio": -9}],
            completed_trial_sharpes=[1.0],
            enabled=False,
        )
        is False
    )


def test_save_shortlist_top_k(tmp_path):
    entries = [
        {
            "params": {"tune_level": "fast", "min_signal_margin": 0.08},
            "report": {
                "summary": {"mean_sharpe": 0.1},
                "criteria": {"acceptance": {"passed": False}},
            },
        },
        {
            "params": {"tune_level": "fast", "min_signal_margin": 0.10},
            "report": {
                "summary": {"mean_sharpe": 0.5},
                "criteria": {"acceptance": {"passed": True}},
            },
        },
    ]
    path = save_shortlist(entries, path=tmp_path / "shortlist.json", top_k=1)
    data = __import__("json").loads(path.read_text(encoding="utf-8"))
    assert len(data["entries"]) == 1
    assert data["entries"][0]["mean_sharpe"] == 0.5
