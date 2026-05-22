# Руководство по GUI IST

## Запуск

```bash
py -3 -m gui.app
# демо без ML/CLI:
py -3 -m gui.app --demo
```

Зависимости: `requirements-gui.txt` (PyQt6, pyqtgraph).

## Панель символа

- Выбор **пары OKX** и **таймфрейма**
- Статус: `OHLCV | feat | bundle | acc:PASS/FAIL` (последний журнал WFO)

## Вкладки

| Вкладка | Назначение |
|---------|------------|
| **Обзор** | Explain: 4 модели, regime, сигнал, HOLD/why_blocked |
| **График** | Свечи OKX; сигналы — в Обзоре и Исполнении |
| **Модели** | manifest bundle, model_keys, веса |
| **Бэктесты** | Журнал WFO, PASS/FAIL, equity по фолдам |
| **Исполнение** | Paper: шаг, серия шагов, сверка с Backtester |
| **Задачи** | Чеклист pipeline + CLI |
| **Конфигурация** | Параметры (merged с эталоном), acceptance-отчёт |

## Задачи (CLI из GUI)

1. **Подготовить символ** — `prepare-symbol` (+ опционально скачать OHLCV)
2. **Признаки** — `build-features`
3. **Тюнинг thesis** — `tune-thesis` (фаза: all / fast / refine / confirm)
4. **Финальное обучение** — `train-final-symbol --force`
5. **Отчёт WFO** — `report-real` с `--use-tuning-best`, `--use-feature-cache`, баров `0` = из YAML

## Конфигурация

- Эталон: `config/reference/thesis_4model_reference.yaml`
- Пара: `baseline_ref` + `orchestration_overrides` в `config/symbols/<slug>.yaml`
- **Сохранить** — пишет только отличия от эталона
- **Отчёт acceptance** — полный WFO в `docs/thesis_<slug>_acceptance_gui.json`

## Сценарий защиты (5 мин)

1. BTC/USDT 1h → **Обзор** (4 prob, regime)
2. **Бэктесты** — последний run, 7/8 или PASS
3. **Конфигурация** — margin, vol filter, эталон
4. Переключить **ETH** — статус pipeline
5. **Исполнение** — paper, 1–5 шагов

Исследование и финальный тюнинг по-прежнему можно вести в CLI; GUI повторяет те же флаги для отчётов.
