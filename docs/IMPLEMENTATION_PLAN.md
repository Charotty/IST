# План реализации: каноническая IST (4 модели + чистый репозиторий)

**Цель:** привести проект в соответствие с формулировкой диплома (ансамбль LGB + XGB + GRU + CNN, regime-adaptive meta, confidence gate, WFO без утечек), **не переписывая** рабочий orchestration-стек.

**Канон:** единственный production/research path — `orchestration` + `model_keys: [lgb, gru, xgb, cnn]`.

**В архив (обсуждение):** `orchestration_tuning_best` (2 модели, другие пороги, `fixed_range`, `use_risk_bridge: false`).

**Статус задач:** ⬜ не начато · 🔄 в работе · ✅ готово

---

## Фаза 0 — Канон конфигурации и архив (1–2 дня)

| ID | Задача | Критерий готовности | Статус |
|----|--------|---------------------|--------|
| 0.1 | Удалить `orchestration_tuning_best` из активного `config.yaml` | В корневом YAML только секция `orchestration` с 4 моделями | ✅ |
| 0.2 | Перенести tuning-best в `config/archive/discussion/orchestration_tuning_best_lgb_xgb.yaml` | Файл + `config/archive/discussion/README.md` с пояснением «эксперимент, не канон» | ✅ |
| 0.3 | Создать **`config/profiles/canonical_4model.yaml`** | Полный профиль: orchestration (4 keys, regime_adaptive), feature_engineering, backtesting, `use_risk_bridge: true` | ✅ |
| 0.4 | Сделать `config.yaml` → re-export или merge: `orchestration` = содержимое canonical | `OrchestratorConfig.from_yaml("config.yaml")` → 4 model_keys | ✅ |
| 0.5 | Обновить `config/README.md` | Описать: canonical vs archive vs `config/symbols/*.yaml` | ✅ |
| 0.6 | Отключить default `use_tuning_best` в CLI | `report-real` / `prepare-symbol` по умолчанию читают **canonical**, не tuning_best | ✅ |

**Файлы:** `config.yaml`, `config/profiles/canonical_4model.yaml`, `config/archive/discussion/*`, `orchestration/real_data_benchmark.py`, `orchestration/__main__.py`, `orchestration/tuning_loop.py` (не писать tuning_best обратно в root yaml).

---

## Фаза 1 — Четыре модели работают end-to-end (3–5 дней)

| ID | Задача | Критерий готовности | Статус |
|----|--------|---------------------|--------|
| 1.1 | **Сбор данных под полный профиль** | Скрипт/CLI: OKX → `data/ohlcv/<slug>.parquet` с датами из canonical | ✅ |
| 1.2 | **MTF в pipeline сбора** | После OHLCV: 15m + 4h → merge на 1h → колонки `rsi_15m`, `adx_4h`, … | ✅ |
| 1.3 | **FeatureManager** на полном ряду | `feature_engineering` с `microstructure.mode: off` (диплом) или `simulated` (ablation) | ✅ |
| 1.4 | **WFO с 4 моделями** | `python -m orchestration from-parquet ... --full-models` без skip TF; все 4 ключа в factory | ✅ |
| 1.5 | **Тест интеграции 4-model** | `tests/test_orchestration_four_models.py`: synthetic или small parquet, все keys predict | ✅ |
| 1.6 | **symbol_pipeline: tune/train на 4 моделях** | `tuning_loop` baseline defaults → `model_keys: [lgb, gru, xgb, cnn]`; symbol yaml не перезаписывает на 2 модели без явного флага | ✅ |
| 1.7 | **manifest / journal** | Прогон canonical записан в `docs/backtest_journal/` с полем `profile: canonical_4model` | ✅ |
| 1.8 | **Ablation (диплом)** | 4 прогона: lgb only → lgb+xgb → 4 equal → 4 + regime_adaptive + decision | ✅ |

**Команда-эталон (после фазы 1):**

```bash
python -m orchestration prepare-symbol BTC/USDT 1h --download --config config/profiles/canonical_4model.yaml
# или по шагам:
python -m data_layer ...
python -m synchronization ...
python -m orchestration from-parquet data/features/BTC-USDT_1h.parquet --config config/profiles/canonical_4model.yaml --full-models
```

---

## Фаза 2 — Убрать путаницу в репозитории (2–3 дня)

| ID | Задача | Критерий готовности | Статус |
|----|--------|---------------------|--------|
| 2.1 | **Корневой `README.md`** | Один вход: что такое IST, канон pipeline, ссылка на этот план и `DIPLOMA_TECHNICAL_ANALYSIS.md` | ⬜ |
| 2.2 | Переписать README слоёв под orchestrator | `data_layer`, …, `gui` — статус ✅/⬜ и ссылка на canonical | 🔄 (май 2026: layer README + `docs/vkr/`) |
| 2.3 | **`models/__init__.py` docstring** | Явно: multi-model path = orchestration; router = legacy | ⬜ |
| 2.4 | **Архив `ist.py`** | `archive/ist_colab_research.py` + строка в root README «не канон» | ⬜ |
| 2.5 | **Исправить `docs/INTEGRATION_GAPS_REFERENCE.md`** | Путь фабрики: `orchestration/model_factory.py` | ✅ |
| 2.6 | **Удалить/пометить мёртвые ссылки** | `walk_forward` deprecated — в README backtesting один путь: `TrainingOrchestrator.walk_forward_backtest` | ⬜ |
| 2.7 | **Терминология regime** | В README meta: бинарный `regime_pred`; `breakout_weights` — roadmap или реализация в фазе 4 | ⬜ |

---

## Фаза 3 — «Желательное» из аудита (3–5 дней)

### 3A. Сквозной pipeline CLI

| ID | Задача | Критерий готовности | Статус |
|----|--------|---------------------|--------|
| 3.1 | `scripts/run_canonical_pipeline.py` или `python -m orchestration pipeline` | Subcommands: `fetch`, `mtf`, `features`, `wfo`, `train-bundle` | ⬜ |
| 3.2 | Единый `--config config/profiles/canonical_4model.yaml` | Все шаги читают один профиль | ⬜ |
| 3.3 | Документ `docs/REPRODUCE_CANONICAL.md` | Copy-paste команды для BTC-USDT 1h | ⬜ |

### 3B. Risk bridge в каноне

| ID | Задача | Критерий готовности | Статус |
|----|--------|---------------------|--------|
| 3.4 | `use_risk_bridge: true` в canonical profile | WFO возвращает `position_sizes`; Backtester с `position_size` | ⬜ |
| 3.5 | Опционально ATR trailing в том же прогоне | Флаг `apply_atr_trailing: true` в профиле | ⬜ |
| 3.6 | Тест: metrics с/без risk bridge | Зафиксировать в ablation appendix | ⬜ |

### 3C. Микроструктура (честность диплома)

| ID | Задача | Критерий готовности | Статус |
|----|--------|---------------------|--------|
| 3.7 | Canonical: `microstructure.mode: off` | FeatureManager не требует OBI для 4-model WFO | ⬜ |
| 3.8 | Отдельный профиль `research_with_simulated_obi.yaml` | Для RL/OBI экспериментов | ⬜ |
| 3.9 | Roadmap live L2 в `feature_engineering/README.md` | Шаги: data_layer WS → l2_adapter → resample 1h | ⬜ |

### 3D. Paper demo (inference + execution)

| ID | Задача | Критерий готовности | Статус |
|----|--------|---------------------|--------|
| 3.10 | `train_final` с bundle на 4 моделях | `artifacts/<slug>/<run_id>/` + manifest `bundle_run_id` | ⬜ |
| 3.11 | `python -m orchestration paper-step --bundle ...` | Один бар: InferenceOrchestrator → PaperBroker | ⬜ |
| 3.12 | GUI desktop (PyQt6) + journal | `python -m gui.app`, вкладки График/Режим/Задачи/Конфигурация | ✅ |

### 3E. Научное сравнение ensemble modes

| ID | Задача | Критерий готовности | Статус |
|----|--------|---------------------|--------|
| 3.13 | Прогон `ensemble_mode: fixed_range` (равные веса 0.25×4) | Таблица vs `regime_adaptive` | ⬜ |
| 3.14 | Скрипт `scripts/run_ablation.py` | JSON/table output для главы 5 ВКР | ⬜ |

---

## Фаза 4 — Regime / breakout (опционально, 1–2 дня)

| ID | Задача | Критерий готовности | Статус |
|----|--------|---------------------|--------|
| 4.1 | Либо убрать `breakout` из `regime_keys` в canonical | Конфиг = коду | ⬜ |
| 4.2 | Либо расширить `DynamicMetaWeighting` на 3 режима | `market_regime` → trend/range/breakout weights | ⬜ |

**Рекомендация для диплома:** 4.1 (быстрее, честнее).

---

## Зависимости между фазами

```text
Фаза 0 (канон + архив)
    ↓
Фаза 1 (4 модели E2E + ablation)
    ↓
Фаза 2 (README)          ← можно параллельно с 1.7–1.8
    ↓
Фаза 3 (CLI, risk, paper, сравнение ensemble)
    ↓
Фаза 4 (breakout — по желанию)
```

---

## Definition of Done (весь план)

Проект считается соответствующим идее диплома, когда:

1. **Единственный активный orchestration-профиль** — 4 модели, `regime_adaptive`, `apply_decision_pipeline: true`.
2. **tuning_best (2 модели)** — только в `config/archive/discussion/`, не в default CLI.
3. **Один воспроизводимый прогон** documented в `docs/REPRODUCE_CANONICAL.md` с journal JSON.
4. **Ablation-таблица** (single → full adaptive) в `docs/reports/`.
5. **README слоёв** согласованы с `orchestration/model_factory.py`.
6. **Сквозной pipeline** одной командой или documented chain ≤ 5 команд.
7. **`ist.py`** в archive, не в корне как «главный файл».

---

## Риски и как закрыть

| Риск | Митигация |
|------|-----------|
| TensorFlow не установлен | `requirements-ml.txt` + в CI/test mark `@pytest.mark.tf` |
| Долгий WFO на 4 DL | `dl_epochs: 3` в canonical; полный тюнинг — ночной прогон |
| Holdout не проходит acceptance | В ВКР: методология ✅, PnL — future work; ablation всё равно валиден |
| symbol_pipeline перезаписывает 2-model yaml | Фаза 1.6: tune только с `--profile canonical` |

---

## Связанные документы

| Файл | Назначение |
|------|------------|
| [DIPLOMA_TECHNICAL_ANALYSIS.md](./DIPLOMA_TECHNICAL_ANALYSIS.md) | Технический разбор для ВКР |
| [INTEGRATION_GAPS_REFERENCE.md](./INTEGRATION_GAPS_REFERENCE.md) | Разрывы интеграции |
| [AUDIT_ISSUES_REFERENCE.md](./AUDIT_ISSUES_REFERENCE.md) | Аудит утечек |
| `config/profiles/canonical_4model.yaml` | Канонический профиль (создаётся в фазе 0) |
| `config/archive/discussion/` | Архив tuning-best |

---

*Последнее обновление плана: 2026-05-17. При закрытии задачи меняйте статус в таблице на ✅.*
