# Ускорение `tune-thesis`

## Где уходит время

Один **trial** ≈ полный WFO:

```text
trials × n_folds × (fit LGB + XGB + GRU + CNN + 2× pipeline + 2× backtest)
```

DL (GRU/CNN) и число **фолдов** — главные рычаги. Журнал и feature cache уже уменьшают накладные расходы.

## Встроенные уровни (default)

Файл: `config/profiles/thesis_tuning.yaml`

| Уровень | Фолды (cap) | DL epochs | Trials | Модели |
|---------|-------------|-----------|--------|--------|
| fast | 8 | 2 | 24 | 4 |
| refine | 12 | 2 | 12 | 4 |
| confirm | все (~18) | 3 | 3 | 4 |

## Профиль **turbo** (~3–6× быстрее fast+refine)

Файл: `config/profiles/thesis_tuning_turbo.yaml`

| Изменение | Эффект |
|-----------|--------|
| `tabular_first_fast_refine: true` | fast/refine только **LGB+XGB** (без GRU/CNN) |
| `max_wfo_folds: 5` / `8` | меньше переобучений на фолд |
| `max_rows: 4000` / `6000` | короче история |
| `walk_forward_step: 600` | меньше окон |
| `dl_epochs: 1` на fast/refine | быстрее DL на confirm |
| `n_trials: 12` / `8` | меньше кандидатов |
| `n_estimators_fast: 50` | быстрее табличные модели |
| Pruner после 2 фолдов | плохие trial обрываются раньше |

```bash
python -m orchestration tune-thesis --symbol BTC/USDT --profile turbo --phase all
```

Только отсев (самое быстрое):

```bash
python -m orchestration tune-thesis --symbol ETH/USDT --profile turbo --phase fast
```

## Другие приёмы без смены YAML

1. **Кэш фичей** (не `--no-feature-cache`); один раз: `build-features`.
2. **Только нужная фаза**: `--phase fast` или `fast` → `refine` → `confirm` по отдельности.
3. **Проект на быстром диске**: `~/IST` в WSL вместо `/mnt/d/IST`.
4. **GPU**: XGB `device: cuda` в yaml; TensorFlow уже на GPU.
5. **Меньше случайных trial**: правка `n_trials` в yaml.
6. **Confirm не в tune**: вручную задать params из shortlist + один `report-real`.

## Ориентиры по времени (GTX 1660 Ti, 1 пара)

| Режим | Порядок |
|-------|---------|
| `all` default | много часов |
| `all` turbo | ~1–3 ч (зависит от ETH/BTC) |
| `fast` turbo | ~20–40 мин |
| `confirm` только | ~30–60 мин |

Финальные цифры диплома — всегда **`report-real`** с `thesis_tuning.yaml` confirm-настройками или сохранённым shortlist.
