# Эталонный профиль 4-model (reference baseline)

## Назначение

Файл **`config/reference/thesis_4model_reference.yaml`** — единственный источник параметров, на которых достигнут **лучший подтверждённый результат** (BTC/USDT 1h, confirm variant B). Его меняют **осознанно и редко** (новый confirm, смена архитектуры).

Все рабочие конфиги пар строятся так:

```text
reference (эталон)
    + orchestration_overrides (только отличия пары)
    = эффективный orchestration_tuning_best
```

## Файлы

| Файл | Роль |
|------|------|
| `config/reference/thesis_4model_reference.yaml` | Эталон (margin 0.06, vol filter 90, 4 model_keys, WFO окна) |
| `config/symbols/BTC-USDT_1h.yaml` | `baseline_ref` + минимальные overrides (`tune_source`) |
| `config/symbols/ETH-USDT_1h.yaml` | `baseline_ref` + overrides под ETH (до своего confirm) |
| `docs/thesis_btc_4model_acceptance.json` | Зафиксированный отчёт WFO (18 фолдов, 7/8 acceptance) |

## Код

- `orchestration.symbols.resolve_tuning_best_block()` — merge эталона и overrides
- `orchestration.symbols.tuning_best_for()` — то, что видят CLI/GUI/inference
- `orchestration.symbols.write_symbol_config(..., store_as_overrides=True)` — при сохранении пишет только отличия от эталона

## Команды (эталон + пара)

```bash
# Отчёт с merged params (как variant B)
python -m orchestration report-real --symbol BTC/USDT \
  --use-tuning-best --use-feature-cache \
  --json-out docs/thesis_btc_4model_acceptance.json

# Тюнинг другой пары → overrides в config/symbols/<slug>.yaml
python -m orchestration tune-thesis --symbol ETH/USDT --phase all
```

## Метрики эталона (BTC, variant B)

См. `docs/thesis_btc_4model_acceptance.json`: mean Sharpe ~2.48, PF ~1.78, recovery ~1.26; **WFE ~0.21** (не проходит `min_wfe: 0.5`). Остальные 7 проверок acceptance — PASS.

Разбор WFE (почему низкий при сильном OOS Sharpe): [`WFE_ANALYSIS.md`](WFE_ANALYSIS.md). Диагностика: `python scripts/analyze_wfe.py`.

## Обновление эталона

1. Прогнать `tune-thesis --phase confirm` и `report-real` с кандидатом.
2. Если результат **лучше** текущего эталона — обновить **только** `thesis_4model_reference.yaml`.
3. Упростить `orchestration_overrides` в symbol YAML (оставить только то, что реально отличается).
4. Обновить `docs/thesis_btc_4model_acceptance.json` и этот файл.

Пары **не копируют** полный блок params — только overrides.
