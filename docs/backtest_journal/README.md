# Журнал результатов бэктеста

Отдельное хранилище метрик WFO на **реальных** OHLCV (не pytest).

| Файл | Назначение |
|------|------------|
| `runs.jsonl` | Краткая запись каждого прогона (append-only) |
| `runs/<run_id>.json` | Полный снимок: фолды, criteria, params |
| `INDEX.md` | Таблица последних прогонов (автогенерация) |

Запись прогона:

```powershell
py -3 -m orchestration report-real --parquet data/ohlcv/BTC-USDT_1h.parquet --max-rows 6000
```

Подбор до критериев (а)/(б):

```powershell
py -3 -m orchestration tune-until --parquet data/ohlcv/BTC-USDT_1h.parquet --max-rows 8000 --max-trials 80 --apply-best
```

Пороги: `config.yaml` → `backtesting.acceptance` / `backtesting.target`.

Справочник метрик: `docs/BACKTESTING_CRITERIA_REFERENCE.md`.

**План достижения целей (а)/(б):** `docs/BACKTESTING_ACTION_PLAN.md`.

**GUI:** таблица прогонов — вкладка «Задачи» → «Журнал WFO» (`gui/api/backtests_api.py`).
