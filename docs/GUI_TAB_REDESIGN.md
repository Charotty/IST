# Пересборка вкладок GUI (3 опорные)

Реализовано: **2026-05-23**.

## Структура

| Верхняя вкладка | Содержимое | Исходники (логика без изменений) |
|-----------------|------------|----------------------------------|
| **График** | Chart + подпанели | `chart_view.py`, `overview_view.py`, `regime_view.py`, `models_view.py` |
| **Задачи** | Pipeline / Журнал / Paper | `jobs_view.py`, `backtests_view.py`, `execution_view.py` |
| **Конфигурация** | YAML + acceptance | `settings_view.py` |

Обёртки: `chart_hub_view.py`, `jobs_hub_view.py`, `main_window.py`.

## Принцип

- Не переписывать понравившиеся экраны — только **компоновать**.
- Старые имена вкладок в `i18n_ru.py` оставлены для совместимости доков/скриптов.

## См. также

- [GUI_LAYOUT_REFERENCE.md](GUI_LAYOUT_REFERENCE.md) — детальные макеты
- [PROJECT_CLI_AND_GUI_COMMANDS.md](PROJECT_CLI_AND_GUI_COMMANDS.md) — матрица CLI
