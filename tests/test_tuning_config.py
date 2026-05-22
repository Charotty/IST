"""4-model tuning config keeps regime-adaptive weights (not flat 0.25)."""

from orchestration.tuning_config import THESIS_4MODEL_SEED, orchestrator_config_from_tuning_params


def test_four_model_regime_weights_not_flat():
    cfg = orchestrator_config_from_tuning_params(THESIS_4MODEL_SEED)
    assert cfg.model_keys == ["lgb", "gru", "xgb", "cnn"]
    assert cfg.ensemble_mode == "regime_adaptive"
    assert cfg.trend_weights["gru"] > cfg.trend_weights["lgb"]
    assert cfg.range_weights["lgb"] > cfg.range_weights["gru"]
    assert abs(sum(cfg.trend_weights.values()) - 1.0) < 1e-6


def test_two_model_still_works():
    p = {
        **THESIS_4MODEL_SEED,
        "model_keys": ["lgb", "xgb"],
        "ensemble_mode": "fixed_range",
    }
    cfg = orchestrator_config_from_tuning_params(p)
    assert set(cfg.model_keys) == {"lgb", "xgb"}
