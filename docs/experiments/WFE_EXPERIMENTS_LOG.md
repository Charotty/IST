# Журнал экспериментов WFE (BTC/USDT 1h, variant B)

Эталон: `config/reference/thesis_4model_reference.yaml` + overrides в `config/symbols/BTC-USDT_1h.yaml`.

## Сводная таблица

| ID | test_window | JSON | mean Sharpe | mean PF | mean ret % | mean WFE | median WFE | recovery | trades | acceptance |
|----|-------------|------|-------------|---------|------------|----------|------------|----------|--------|------------|
| **baseline** | 170 | `thesis_btc_4model_acceptance.json` | **2.48** | 1.78 | **0.74** | 0.21 | ~0.07 | **1.26** | 381 | **7/8** (WFE) |
| exp_t280 | 280 | `thesis_btc_wfe_exp_t280.json` | 0.86 | 1.31 | −0.09 | 0.03 | — | 0.93 | 554 | 5/8 |
| **exp_t140** | 140 | `thesis_btc_wfe_exp_t140.json` | 1.49 | 1.74 | 0.40 | **0.59** | ~0 | 0.98 | 278 | **7/8** (recovery) |
| **ETH** | 170 | `thesis_eth_4model_acceptance.json` | **0.75** | 1.47 | 0.20 | **0.075** | ~0 | 0.59 | 297 | **6/8** (WFE, recovery) |

Проверка: `parquet` в JSON = `ETH-USDT_1h.parquet`. Первый прогон без `--parquet` писал BTC в метаданные, но фичи уже брались из кэша **ETH** (`--symbol` + `--use-feature-cache`) — OOS совпал с повтором.

`tune_source`: BTC baseline `variant_b_vol_filter_90`; BTC exp `wfe_exp_test280` / `wfe_exp_test140`; ETH `eth_pending_confirm` + overrides (horizon 24, margin 0.08, vol 0, max_pos 1.0).

## Выводы

- **t280** — хуже baseline по всем ключевым OOS-метрикам; не использовать.
- **t140** — формально **PASS `min_wfe` (0.59 ≥ 0.5)**; mean Sharpe ниже baseline (1.49 vs 2.48); **FAIL `min_recovery_factor` (0.98 < 1.0)** на ~2%.
- Mean WFE на t140 сильно тянут **выбросы** (фолды 8, 13: WFE > 3 при коротком OOS и экстремальной годовизации CAGR); **median WFE ≈ 0** — в ВКР указывать mean и median вместе.

## Рекомендация для диплома

| Цель | Конфиг |
|------|--------|
| Сильный OOS Sharpe / PF | **test_window 170** (эталон reference, override только `tune_source`) |
| Формальный PASS по WFE + пояснение IS/OOS | t140 как **чувствительный анализ** (§3.11), не подменять основной ряд без обоснования |
| Полный **8/8** acceptance | t140 + доработка recovery (микро-подгон или повтор с seed); либо обосновать 0.98 ≈ 1.0 |

## Команды

```bash
python -m orchestration report-real --symbol BTC/USDT \
  --use-tuning-best --use-feature-cache \
  --json-out docs/thesis_btc_wfe_exp_t140.json

python scripts/compare_wfe_reports.py docs/thesis_btc_4model_acceptance.json docs/thesis_btc_wfe_exp_t140.json
```
