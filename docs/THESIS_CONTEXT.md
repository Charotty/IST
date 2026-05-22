# Thesis work context — IST 4-model training & validation

Living document for the diploma: environment, status, and **two active engineering tasks**.

---

## Goal (updated)

Prove that the system **trains and is tested end-to-end with high confidence** on **multiple instruments**, using the **full project design**:

- **4 model keys:** `lgb`, `gru`, `xgb`, `cnn`
- **Regime-adaptive or validated ensemble mode** (not a reduced 2-model shortcut as the main claim)
- **Leakage-safe walk-forward (WFO)** + automated **acceptance** criteria
- **Per-symbol** calibration (BTC, ETH, …) with the **same architecture**

Success for the thesis = **reproducible pipeline** + **acceptance-level OOS metrics** on **≥2 pairs**, documented in `docs/backtest_journal/` and JSON reports.

---

## Environment

| Item | Status |
|------|--------|
| OS | WSL2 on Windows |
| Project path | `/mnt/d/IST` (faster copy at `~/IST` optional) |
| Python | 3.11 venv `.venv` |
| GPU | NVIDIA GTX 1660 Ti, CUDA via TensorFlow `[and-cuda]` |
| Data | `data/ohlcv/BTC-USDT_1h.parquet`, `ETH-USDT_1h.parquet` |
| Profile | `config/profiles/canonical_4model.yaml` |
| Criteria | `config.yaml` → `backtesting.acceptance` (thesis bar) |

```bash
cd /mnt/d/IST && source .venv/bin/activate
export TF_CPP_MIN_LOG_LEVEL=2
```

---

## What already works

| Check | Status |
|-------|--------|
| `nvidia-smi` in WSL | OK |
| TensorFlow sees GPU | OK |
| Full pipeline 4 models on GPU (GRU/CNN XLA) | OK |
| WFO + metrics + journal | OK |
| `thesis_push_4model` — 18 folds, ~30 min/trial (full WFO) | OK |
| Unit tests + `orchestration/tuning_config` (regime weights) | OK |
| **2-model** historical acceptance (sanity) | OK — journal `20260516T130409Z_23a45cab` (Sharpe 0.55, PF 2.12, WFE 0.98) |

---

## What is not achieved yet

| Item | Status |
|------|--------|
| **`acceptance_passed: true` with 4 models** on BTC | Not yet — best push Sharpe **~0.33** (need **> 0.5**) |
| Strong **WFE + Sharpe** together on 4-model runs | Often PF/WFE OK, Sharpe fails |
| ETH (and other pairs) with same methodology | Not run after latest presets |
| Final thesis JSON + symbol YAML locked to PASS run | Pending |

---

## Implementation map (what to re-study)

| Layer | Module | Role in training time / quality |
|-------|--------|----------------------------------|
| Features | `feature_engineering/feature_engine.py` | Once per OHLCV load; cache helps multi-trial |
| Model factory | `orchestration/model_factory.py` | Builds 4 models; **`dl_epochs`**, batch 64 |
| WFO loop | `orchestration/training_orchestrator.py` | **Per fold:** fit ×4 + pipeline ×2 + backtest |
| Tuning | `orchestration/tuning_loop.py` → `run_single_trial` | One full WFO per hyperparameter trial |
| Config merge | `orchestration/tuning_config.py` | **Regime weights** vs flat 0.25; presets |
| Criteria | `backtesting/criteria_evaluator.py` | acceptance = 8 checks |
| Per-symbol | `config/symbols/<slug>.yaml` | Best params after PASS |

**Cost driver:**  
`trials × n_folds × (train LGB + XGB + GRU + CNN + inference pipeline)`.

---

## Two tasks (current focus)

### Task 1 — Speed up training / tuning

**Objective:** Find hyperparameters in **hours, not days**, without breaking the 4-model contract.

**Levers (implemented in repo):**

| Lever | Where | Effect |
|-------|-------|--------|
| `THESIS_FAST_PRESET` | `orchestration/tuning_config.py` | `dl_epochs=2`, `max_wfo_folds=8`, `step=500` |
| `--mode fast` | `scripts/thesis_push_4model.py` | Applies fast preset |
| `--max-rows 6000` | CLI / preset | Less data |
| Larger `walk_forward_step` | params | Fewer folds |
| Project on `~/IST` | filesystem | Faster I/O than `/mnt/d/` |
| `prepare-symbol --max-trials 5` | CLI | Not 30 full-history trials |

**Commands:**

```bash
# Fast search (~8 folds × 2 DL epochs — use for exploration only)
python scripts/thesis_push_4model.py \
  --parquet data/ohlcv/BTC-USDT_1h.parquet \
  --mode fast

# Combined: fast WFO cap + quality-oriented ensemble defaults
python scripts/thesis_push_4model.py \
  --parquet data/ohlcv/BTC-USDT_1h.parquet \
  --mode fast+quality
```

**Note:** `max_wfo_folds > 0` is for **tuning only**. Final thesis report must use `max_wfo_folds: 0` and `n_folds ≥ 5`.

**Further ideas (not all coded):**

- Pre-save feature parquet per symbol; trials read features only
- Tabular-only tune → then one 4-model confirm run
- Mixed precision / larger batch on GPU (GRU/CNN)
- Parallel trials (not in repo today)

---

### Task 2 — Strong results (acceptance + transfer to other pairs)

**Objective:** High OOS coefficients (Sharpe, PF, WFE) on **4 models**, reproducible on **BTC and ETH**.

**Levers (implemented):**

| Lever | Rationale |
|-------|-----------|
| `THESIS_QUALITY_PRESET` | `fixed_range` + acceptance-like thresholds (2-model journal) but **4 keys** |
| `--mode quality` | Default for `thesis_push_4model.py` |
| `thesis_push_4model_phase2.py` | Extra grid around `fixed_range` |
| `min_signal_margin` 0.08–0.12 | Filters weak signals → Sharpe |
| `volatility_filter_percentile: 90` | Fewer bad bars |
| Per-symbol YAML | `config/symbols/BTC-USDT_1h.yaml`, `ETH-USDT_1h.yaml` |

**Commands:**

```bash
# Quality push (full WFO, 8000 bars)
python scripts/thesis_push_4model.py \
  --parquet data/ohlcv/BTC-USDT_1h.parquet \
  --mode quality \
  --write-symbol-yaml

python scripts/thesis_push_4model_phase2.py \
  --parquet data/ohlcv/BTC-USDT_1h.parquet \
  --max-rows 8000 \
  --write-symbol-yaml

# Final report after YAML shows PASS
python -m orchestration report-real \
  --parquet data/ohlcv/BTC-USDT_1h.parquet \
  --max-rows 8000 \
  --use-tuning-best \
  --symbol BTC/USDT \
  --timeframe 1h \
  --dl-epochs 3 \
  --json-out docs/thesis_btc_4model_acceptance.json

# ETH — same workflow
python scripts/thesis_push_4model.py --symbol ETH/USDT --mode quality
```

**Acceptance thresholds** (`config.yaml`):

- `min_folds ≥ 5`, `min_oos_sharpe > 0.5`, `min_oos_profit_factor ≥ 1.2`, `min_wfe ≥ 0.5`, positive mean return, DD ≥ −35%, `min_trade_events ≥ 30`.

**Quality strategy (order of work):**

1. **Fast mode** → shortlist 3–5 param patches (Task 1).  
2. **Quality / phase2** → full WFO on shortlist (Task 2).  
3. **report-real** + journal run_id for thesis tables.  
4. **Repeat per symbol** (ETH); compare in one table “architecture same / params per pair”.  
5. **Ablation** — `scripts/run_ablation.py` for §3.11 (4 adaptive vs static vs LGB+XGB).

---

## Recommended workflow (study → achieve)

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Re-read tuning_config.py + training_orchestrator WFO     │
│ 2. Task 1: --mode fast  →  shortlist candidates             │
│ 3. Task 2: --mode quality / phase2  →  acceptance PASS      │
│ 4. report-real --use-tuning-best  →  thesis JSON + run_id     │
│ 5. ETH + ablation + GUI journal screenshots                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Thesis artifacts checklist

- [ ] `docs/thesis_btc_4model_acceptance.json` — ACCEPTANCE PASS, 4 models  
- [ ] `docs/thesis_eth_4model_acceptance.json`  
- [ ] `docs/reports/ablation_thesis_8k.json`  
- [ ] `config/symbols/*.yaml` — only after PASS  
- [ ] `docs/backtest_journal/` — run_id cited in text  
- [ ] §3.11 tables: 4-model adaptive vs baselines  

---

## Related docs

- [`THESIS_ACCELERATION_IMPLEMENTATION_PLAN.md`](THESIS_ACCELERATION_IMPLEMENTATION_PLAN.md) — **план реализации** §1–§4 (ускорение + качество); код по нему пока не ведётся
- `docs/THESIS_4MODEL_STEPS.md` — step-by-step CLI  
- `docs/BACKTESTING_CRITERIA_REFERENCE.md` — metric meanings  
- `docs/3_11_SYSTEM_TESTING.md` — chapter template  

---

*Last updated: session context — Task 1 (speed) + Task 2 (strong metrics) active.*
