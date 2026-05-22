# Диплом: пошаговый контур (4 модели, acceptance, BTC + ETH)

Цель: **`acceptance_passed: true`** (8/8) с 4 моделями на **≥5** WFO-фолдах; **BTC** — эталон в reference YAML (сейчас **7/8**, FAIL только WFE). **ETH** — те же 4 ключа, overrides после `tune-thesis`.

Критерии — `config.yaml` → `backtesting.acceptance` (см. `docs/BACKTESTING_CRITERIA_REFERENCE.md`).

---

## Подготовка (один раз)

```bash
cd /mnt/d/IST   # или ~/IST
source .venv/bin/activate
pip install -r requirements.txt
export TF_CPP_MIN_LOG_LEVEL=2
```

Проверка:

```bash
python scripts/thesis_4model_steps.py --step 0
```

Нужны файлы:

- `data/ohlcv/BTC-USDT_1h.parquet`
- `data/ohlcv/ETH-USDT_1h.parquet` (или `--download` на шаге 5)

Конфигурация (эталон + пары):

- **Эталон (менять один раз):** `config/reference/thesis_4model_reference.yaml` — лучший BTC confirm (variant B)
- **BTC:** `config/symbols/BTC-USDT_1h.yaml` — `baseline_ref` + overrides (`tune_source`)
- **ETH:** `config/symbols/ETH-USDT_1h.yaml` — `baseline_ref` + свои overrides до confirm

См. [`THESIS_REFERENCE.md`](THESIS_REFERENCE.md). CLI/GUI читают merged params через `tuning_best_for()`.

---

## Все шаги одной командой

```bash
python scripts/thesis_4model_steps.py --step all
```

Или по отдельности:

| Шаг | Команда | Что даёт |
|-----|---------|----------|
| 0 | `--step 0` | Python, TF, GPU, parquet |
| 1 | `--step 1` | pytest (быстро) |
| 2 | `--step 2` | Baseline WFO 4 модели с семенем |
| 3 | `--step 3` | Узкий подбор → acceptance |
| 4 | `--step 4` | Ablation для табл. 3.24 |
| 5 | `--step 5` | ETH: push + report |
| 6 | `--step 6` | Список артефактов |

---

## Шаг 2 — baseline (после семени в YAML)

```bash
python -m orchestration report-real \
  --parquet data/ohlcv/BTC-USDT_1h.parquet \
  --max-rows 8000 \
  --use-tuning-best \
  --symbol BTC/USDT \
  --timeframe 1h \
  --dl-epochs 3 \
  --json-out docs/thesis_seed_BTC-USDT_1h.json
```

Ожидание: `n_folds` ≥ 5. Если **ACCEPTANCE FAIL** только по WFE/PF — шаг 3.

---

## Две задачи (актуально)

| Задача | Режим | Команда |
|--------|-------|---------|
| **1 — скорость** | `--mode fast` | короткий WFO (`max_wfo_folds=8`, `dl_epochs=2`) |
| **2 — качество** | `--mode quality` | полный WFO, `fixed_range` + 4 модели |

Подробно: `docs/THESIS_CONTEXT.md`.

## Шаг 3 — узкий подбор (4 модели)

```bash
python scripts/thesis_push_4model.py \
  --parquet data/ohlcv/BTC-USDT_1h.parquet \
  --max-rows 8000
```

YAML пишется **только при PASS** (`--write-symbol-yaml`). Не используйте `--write-best-failed`, если не хотите перезаписать семя.

Если FAIL по Sharpe (часто `< 0.5` при regime_adaptive):

```bash
python scripts/thesis_push_4model_phase2.py \
  --parquet data/ohlcv/BTC-USDT_1h.parquet \
  --max-rows 8000 \
  --write-symbol-yaml
```

Фаза 2 стартует с `ensemble_mode: fixed_range` (как в 2-model acceptance), но **все 4 модели** остаются в `model_keys`.

При **`*** ACCEPTANCE PASSED ***`** лучшие параметры записываются в `config/symbols/BTC-USDT_1h.yaml`.

Повтор финального отчёта:

```bash
python -m orchestration report-real \
  --parquet data/ohlcv/BTC-USDT_1h.parquet \
  --max-rows 8000 \
  --use-tuning-best \
  --symbol BTC/USDT \
  --timeframe 1h \
  --dl-epochs 3 \
  --json-out docs/thesis_btc_4model_acceptance.json
```

**Для диплома сохраните:** `run_id` из журнала, JSON, блок ACCEPTANCE (PASS).

---

## Шаг 4 — ablation (таблица «полная логика ИТС»)

```bash
python scripts/run_ablation.py \
  --parquet data/ohlcv/BTC-USDT_1h.parquet \
  --max-rows 8000 \
  --dl-epochs 3 \
  --out docs/reports/ablation_thesis_8k.json
```

В тексте: `four_regime_adaptive` vs `four_equal` vs `lgb_xgb`.

---

## Шаг 5 — вторая пара (ETH)

```bash
# если нет parquet:
python -m orchestration prepare-symbol ETH/USDT 1h --download --max-trials 3 --skip-final

python scripts/thesis_push_4model.py --symbol ETH/USDT --timeframe 1h --max-rows 8000 --write-symbol-yaml

python -m orchestration report-real \
  --parquet data/ohlcv/ETH-USDT_1h.parquet \
  --max-rows 8000 \
  --use-tuning-best \
  --symbol ETH/USDT \
  --timeframe 1h \
  --dl-epochs 3 \
  --json-out docs/thesis_eth_4model_acceptance.json
```

---

## Шаг 6 — артефакты и GUI (опционально)

```bash
python -m orchestration prepare-symbol BTC/USDT 1h --max-trials 3
python -m orchestration explain --symbol BTC/USDT --timeframe 1h
```

Журнал: `docs/backtest_journal/runs.jsonl`, индекс `docs/backtest_journal/INDEX.md`.

---

## Что положить в диплом

| Артефакт | Путь |
|----------|------|
| BTC acceptance JSON | `docs/thesis_btc_4model_acceptance.json` |
| ETH JSON | `docs/thesis_eth_4model_acceptance.json` |
| Ablation | `docs/reports/ablation_thesis_8k.json` |
| Per-symbol конфиг | `config/symbols/*.yaml` |
| Run id | `docs/backtest_journal/` |

**Тезис:** единая архитектура 4 моделей + regime-adaptive; калибровка per-symbol; acceptance на OOS WFO.

---

## Ускорение

- `--max-rows 8000` (не полная история на этапе подбора)
- `--dl-epochs 2` в push (быстрее GRU/CNN)
- `walk_forward_step: 360` (уже в семени)
- Не запускать `prepare-symbol --max-trials 30` до успешного `thesis_push_4model.py`

---

## Исправление в коде (важно)

Раньше `orchestrator_config_from_params` ставил **равные веса 0.25** на 4 модели. Для диплома используется `orchestration/tuning_config.py`: canonical **regime_adaptive** веса (GRU в trend, LGB в range и т.д.).
