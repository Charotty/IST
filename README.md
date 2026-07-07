# Intelligent Trading System (IST)

IST - исследовательская и прикладная платформа для алгоритмической торговли: от загрузки рыночных данных и построения признаков до walk-forward backtesting, подготовки артефактов моделей и десктопного мониторинга через GUI.

## Назначение проекта

Проект объединяет единый пайплайн:

- загрузка исторических OHLCV (OKX);
- генерация признаков (в т.ч. multi-timeframe);
- оркестрация ансамбля моделей (`lgb`, `gru`, `xgb`, `cnn`) с regime-aware мета-взвешиванием;
- бэктест/валидация по WFO-критериям;
- подготовка и просмотр артефактов;
- работа через CLI и PyQt6 GUI.

## Архитектура (кратко)

Основной контракт вычислений:

`features -> regime -> model_probs -> meta_weighting -> decision -> risk -> report/artifacts`

Ключевые модули:

- `data_layer/` - загрузка OHLCV и сохранение в `data/`;
- `synchronization/` - синхронизация multi-timeframe рядов;
- `feature_engineering/` - расчет и сохранение признаков;
- `orchestration/` - основной CLI, WFO, symbol pipeline, отчеты и проверка конфигов;
- `models/`, `meta_learning/`, `decision/`, `risk_management/` - модели и логика принятия решений;
- `backtesting/` - оценки качества, критерии и журнал результатов;
- `gui/` - десктопное приложение на PyQt6;
- `config/` - профили конфигураций и per-symbol overrides;
- `docs/` - эксплуатационная и исследовательская документация.

## Требования

- Python `3.11+` (рекомендуется 3.11 для совместимости с GUI/ML стеком).
- ОС: Windows/Linux/macOS (GUI сценарий ориентирован на desktop-окружение с PyQt6).
- Для базового пайплайна: зависимости из `requirements.txt`.
- Для GUI: дополнительно `requirements-gui.txt`.
- Для тюнинга (опционально): `requirements-tune.txt`.

## Установка и окружение

```powershell
cd D:\IST
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-gui.txt
pip install -r requirements-tune.txt
```

Если GUI не нужен, шаг с `requirements-gui.txt` можно пропустить.

## Быстрый старт

1) Проверить CLI и конфиг:

```powershell
python -m orchestration smoke
python -m orchestration validate-config --config config/profiles/canonical_4model.yaml
```

2) Подготовить символ (данные -> признаки -> артефакты):

```powershell
python -m orchestration prepare-symbol BTC/USDT 1h --download --config config/profiles/canonical_4model.yaml
```

3) Сформировать WFO-отчет:

```powershell
python -m orchestration report-real --symbol BTC/USDT --timeframe 1h --use-tuning-best --use-feature-cache --max-rows 0
```

## Основные сценарии запуска

### CLI

Главная точка входа:

```powershell
python -m orchestration --help
```

Полезные команды:

```powershell
# Проверка конфига
python -m orchestration validate-config --config config.yaml

# Построение признаков по символу
python -m orchestration build-features --symbol BTC/USDT --timeframe 1h

# Подробное объяснение текущего состояния модели
python -m orchestration explain --symbol BTC/USDT --timeframe 1h --window 256

# Отчет WFO и критерии приемки
python -m orchestration report-real --symbol BTC/USDT --timeframe 1h --json-out docs/e2e_last_metrics.json
```

Дополнительно:

- `python -m data_layer` - загрузка OHLCV в parquet/csv;
- `python -m feature_engineering` - построение признаков из готового parquet;
- `python -m synchronization` - синхронизация MTF-рядов.

### GUI

Запуск десктоп-приложения:

```powershell
python -m gui.app
```

Демо-режим:

```powershell
python -m gui.app --demo
```

В GUI доступны вкладки мониторинга графика/режима, запуск pipeline-задач и просмотр журналов бэктеста.

## Структура каталогов

```text
IST/
├── backtesting/          # WFO, критерии, журнал результатов
├── config/               # профили и YAML-конфиги
├── data/                 # локальные данные OHLCV/features
├── data_layer/           # загрузка данных (OKX, storage)
├── feature_engineering/  # вычисление признаков
├── orchestration/        # основной pipeline и CLI
├── gui/                  # PyQt6 desktop приложение
├── models/               # модели и связанный ML код
├── docs/                 # документация, гайды, отчеты
├── tests/                # unit/integration тесты
├── requirements.txt
├── requirements-gui.txt
└── requirements-tune.txt
```

## Тестирование

Базовый прогон:

```powershell
pytest tests/ -m "not integration" -q
```

Интеграционные сценарии (дольше и тяжелее):

```powershell
pytest tests/test_orchestration_real_models.py -m integration
pytest tests/test_orchestration_four_models.py -m integration
```

## Конфиги и данные

- Основной конфиг: `config.yaml`.
- Рекомендованный профиль оркестрации: `config/profiles/canonical_4model.yaml`.
- Символьные override-конфиги: `config/symbols/*.yaml`.
- Локальные OHLCV обычно в `data/ohlcv/`, признаки в `data/features/`.
- Для чувствительных значений используйте `.env` и переменные окружения (пример: ключи OKX).

## Артефакты, журналирование и логи

- `artifacts/` - bundle-артефакты моделей, manifest и результаты прогонов.
- `docs/backtest_journal/` - журнал запусков WFO/бэктестов и агрегированные метрики.
- Файлы сборки GUI (PyInstaller): `build/`, `dist/`.
- Логи приложения могут формироваться в файлы (например, `trading_system.log`) согласно настройкам `config.yaml`.

## Troubleshooting

- `ModuleNotFoundError`/`ImportError`: убедитесь, что активировано правильное `.venv` и установлены зависимости.
- GUI не стартует с ошибкой PyQt6: выполните `pip install -r requirements-gui.txt`.
- `validate-config` возвращает ошибки: проверьте согласованность `model_keys` и весов в `orchestration` секции YAML.
- Пустой/слабый `report-real`: проверьте наличие parquet, корректность интервала данных и параметры `--max-rows`, `--use-tuning-best`.
- Медленные тесты: исключайте `integration` маркер для локальной быстрой проверки.

## Статус и roadmap

Текущий статус:

- CLI pipeline и GUI desktop доступны и активно используются;
- базовый live-путь ограничен, основной фокус - research/paper flow и воспроизводимый backtesting.

Ближайшие направления:

- укрепление live-режима и интеграций исполнения;
- дальнейшая автоматизация валидаций и отчетности;
- улучшение UX GUI для длинных задач и мониторинга.
