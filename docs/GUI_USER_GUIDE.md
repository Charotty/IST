# Руководство по GUI IST

Полный список CLI-команд и чеклист «команда → вкладка GUI»: [`PROJECT_CLI_AND_GUI_COMMANDS.md`](PROJECT_CLI_AND_GUI_COMMANDS.md).

**Расположение всех элементов на каждом экране:** [`GUI_LAYOUT_REFERENCE.md`](GUI_LAYOUT_REFERENCE.md) (TOC, сценарии, empty-states, связь с CLI §7, UX-аномалии).

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
| **График** | Live OKX / IST; подпанели **Решение**, **Bundle** |
| **Режим** | Таблица trend/range (отдельный экран) |
| **Задачи** | **Pipeline** (CLI) · **Журнал WFO** · **Paper** |
| **Конфигурация** | Параметры (merged с эталоном), acceptance-отчёт |

**Пошаговая проверка:**

- [`GUI_VERIFICATION_PLAN.md`](GUI_VERIFICATION_PLAN.md) — только GUI  
- [`GUI_VERIFICATION_PLAN_WSL.md`](GUI_VERIFICATION_PLAN_WSL.md) — **GUI + команды WSL на каждый шаг** (обучение, report-real, быстрый контур)

Структура вкладок: [`GUI_TAB_REDESIGN.md`](GUI_TAB_REDESIGN.md).

## Задачи (CLI из GUI)

**Основные:** prepare-symbol, build-features, tune-thesis, train-final, report-real.

**Дополнительные:** smoke, validate-config, tune-until, from-parquet (выбор файла), regime-history, list-symbols, manifest-show.

## Конфигурация

- Эталон: `config/reference/thesis_4model_reference.yaml`
- Пара: `baseline_ref` + `orchestration_overrides` в `config/symbols/<slug>.yaml`
- **Сохранить** — пишет только отличия от эталона
- **Отчёт acceptance** — полный WFO в `docs/thesis_<slug>_acceptance_gui.json`

## Сценарий защиты (5 мин)

1. BTC/USDT 1h → **График** → подпанель **Решение** (4 prob, regime)
2. **Задачи** → **Журнал WFO** — последний run, PASS/FAIL
3. **Конфигурация** — margin, vol filter, эталон
4. Переключить **ETH** — статус pipeline на панели символа
5. **Задачи** → **Paper** — 1–5 шагов

Исследование и финальный тюнинг по-прежнему можно вести в CLI; GUI повторяет те же флаги для отчётов.
