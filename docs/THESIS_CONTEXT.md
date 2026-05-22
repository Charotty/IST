# Thesis work context — IST 4-model

Living document: environment, **reference baseline**, per-symbol overrides, status.

---

## Goal

End-to-end **4 models** (`lgb`, `gru`, `xgb`, `cnn`), regime-adaptive ensemble, leakage-safe WFO, **acceptance** metrics, **≥2 instruments** (BTC + ETH methodology).

**Reference baseline (change once):**  
[`config/reference/thesis_4model_reference.yaml`](../config/reference/thesis_4model_reference.yaml) — best confirm BTC variant B.  
Details: [`THESIS_REFERENCE.md`](THESIS_REFERENCE.md).

**Per-symbol:** `baseline_ref` + `orchestration_overrides` in `config/symbols/*.yaml`.

---

## Environment

| Item | Value |
|------|--------|
| Path | `/mnt/d/IST` or `D:\IST` |
| Python | 3.11, `.venv` |
| GPU | GTX 1660 Ti, TF `[and-cuda]` |
| Profile | `config/profiles/canonical_4model.yaml` |
| Tune profile | `config/profiles/thesis_tuning.yaml` |
| Acceptance | `config.yaml` → `backtesting.acceptance` (8 checks) |

```bash
cd /mnt/d/IST && source .venv/bin/activate
export TF_CPP_MIN_LOG_LEVEL=2
```

---

## Status (2026-05)

| Item | Status |
|------|--------|
| 4-model pipeline + GPU | OK |
| `tune-thesis` fast/refine/confirm | OK |
| Feature cache + `build-features` | OK |
| **BTC reference metrics** (variant B) | OK — see `docs/thesis_btc_4model_acceptance.json` |
| **Acceptance 8/8 on 4 models** | **7/8** — only `min_wfe` (~0.21 vs 0.5) |
| ETH confirm + JSON | Pending — overrides in `ETH-USDT_1h.yaml` |
| GUI ↔ CLI report flags | OK — Jobs/Settings |
| Reference + overrides config layout | OK |

### BTC variant B (effective = reference + `tune_source` override)

- `min_signal_margin: 0.06`, `volatility_filter_percentile: 90`
- 18 folds, mean Sharpe ~2.48, PF ~1.78, recovery ~1.26

---

## Commands

```bash
python -m orchestration build-features --symbol BTC/USDT --timeframe 1h
python -m orchestration tune-thesis --symbol BTC/USDT --phase all
python -m orchestration report-real --symbol BTC/USDT --use-tuning-best --use-feature-cache \
  --json-out docs/thesis_btc_4model_acceptance.json
python -m orchestration train-final-symbol --symbol BTC/USDT --timeframe 1h --force
py -3 -m gui.app
```

ETH: same with `--symbol ETH/USDT`; tune confirm → update **overrides only** in `config/symbols/ETH-USDT_1h.yaml`.

---

## Docs map

| Doc | Content |
|-----|---------|
| [`THESIS_REFERENCE.md`](THESIS_REFERENCE.md) | Reference baseline contract |
| [`THESIS_AND_SYSTEM_ROADMAP.md`](THESIS_AND_SYSTEM_ROADMAP.md) | Unified change plan |
| [`THESIS_ACCELERATION_IMPLEMENTATION_PLAN.md`](THESIS_ACCELERATION_IMPLEMENTATION_PLAN.md) | Speed/quality implementation (done) |
| [`THESIS_4MODEL_STEPS.md`](THESIS_4MODEL_STEPS.md) | CLI steps |
| [`GUI_USER_GUIDE.md`](GUI_USER_GUIDE.md) | Desktop UI |

---

## Implementation map

| Layer | Module |
|-------|--------|
| Reference merge | `orchestration/symbols.py` |
| WFO / tune | `thesis_tuning.py`, `tuning_loop.py`, `real_data_benchmark.py` |
| Criteria | `backtesting/criteria_evaluator.py` |
| GUI | `gui/api/`, `gui/app/` |
