# План проверки IST: GUI + команды WSL

Пошаговый маршрут: **что нажать в GUI** и **что прогнать в WSL**, чтобы быстрее проверить функциональность и отображение.

Связанные файлы:

- [GUI_VERIFICATION_PLAN.md](GUI_VERIFICATION_PLAN.md) — только GUI, без команд
- [PROJECT_CLI_AND_GUI_COMMANDS.md](PROJECT_CLI_AND_GUI_COMMANDS.md) — полный справочник CLI
- [GUI_LAYOUT_REFERENCE.md](GUI_LAYOUT_REFERENCE.md) — где что на экране

---

## Как работать в паре «WSL + GUI»

```text
Терминал WSL (Ubuntu)     →  python -m orchestration …  →  файлы на диске
Окно Windows              →  py -3 -m gui.app           →  отображение
```

1. Репозиторий один: Windows `D:\IST` = WSL `/mnt/d/IST`.
2. После каждой команды в WSL — в GUI: **Toolbar → Обновить** или откройте нужную вкладку (она сама вызовет `refresh`).
3. GUI запускайте **без** `--demo`.
4. Для ускорения обучения ниже есть блок **«Быстрый контур»** (`--dl-epochs 1`, `max-rows 8000`, фаза `fast`).

### Однократная подготовка WSL

```bash
cd /mnt/d/IST
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-gui.txt
export TF_CPP_MIN_LOG_LEVEL=2
# опционально GPU:
# export CUDA_VISIBLE_DEVICES=0
```

Проверка окружения:

```bash
python -m orchestration smoke
python -m orchestration validate-config --config config/profiles/canonical_4model.yaml
```

Запуск GUI (отдельное окно **PowerShell** или cmd на Windows):

```powershell
cd D:\IST
py -3 -m gui.app
```

---

## Переменные для копирования

```bash
export SYM="BTC/USDT"
export TF="1h"
export SLUG="BTC-USDT_1h"
```

---

## Шаг 0 — GUI живой, без обучения

### GUI

| Действие | Ожидание |
|----------|----------|
| Toolbar: **BTC/USDT**, **1h** | Статус пар OKX загружен |
| Вкладка **График** | OKX Live, свечи, статус «обновлено …» |
| Подвкладка **Решение** | Карточка (может быть ошибка без bundle — нормально) |
| Вкладка **Режим** → **Загрузить** | Таблица (если есть OHLCV) |

### WSL (опционально — только данные)

```bash
cd /mnt/d/IST && source .venv/bin/activate
ls -la data/ohlcv/${SLUG}.parquet 2>/dev/null || echo "нет OHLCV — шаг 1"
```

Скачать OHLCV с OKX (если файла нет):

```bash
python -m data_layer -s BTC/USDT -t 1h --from 2022-01-01 --to 2026-01-01
ls -lh data/ohlcv/BTC-USDT_1h.parquet
```

### GUI после WSL

Toolbar **Обновить** → индикатор **OHLCV** должен стать зелёным/жёлтым.

---

## Шаг 1 — Конфигурация пары

### GUI

| Действие | Ожидание |
|----------|----------|
| **Конфигурация** | Форма: `direction_threshold`, `min_signal_margin`, `dl_epochs`… |
| **Тест inference** | Текст с signal / P(up) (нужен bundle) |
| **Сохранить** | «Сохранено», файл `config/symbols/BTC-USDT_1h.yaml` |

### WSL

```bash
cd /mnt/d/IST && source .venv/bin/activate
python -m orchestration validate-config --config config.yaml
# merged config для пары (stdout JSON):
python -c "
import json
from orchestration.symbols import merged_config
print(json.dumps(merged_config('BTC/USDT', '1h'), indent=2, default=str)[:2000])
"
```

### Проверка файла

```bash
test -f config/symbols/BTC-USDT_1h.yaml && head -30 config/symbols/BTC-USDT_1h.yaml
```

### GUI после WSL

**График** → **Решение** (обновится после Save; или Toolbar **Обновить**).

---

## Шаг 1b — Подготовить символ (всё в одной кнопке GUI)

Эквивалент **Задачи** → Pipeline → **Подготовить символ** (+ опционально ☑ **Скачать OHLCV**).

### GUI

| Действие | Ожидание |
|----------|----------|
| ☑ **Скачать OHLCV** | download в логе CLI |
| **Подготовить символ** | Долго: features → tune → train → manifest |
| Нижний лог | JSONL `active/*.jsonl` (отдельно от «Лог CLI») |

### WSL — с загрузкой (долго, как полный pipeline)

```bash
cd /mnt/d/IST && source .venv/bin/activate
export SYM="BTC/USDT" TF="1h"

python -m orchestration prepare-symbol "$SYM" "$TF" \
  --download \
  --config config/profiles/canonical_4model.yaml \
  --max-trials 10
```

Без финального обучения (только данные + tune):

```bash
python -m orchestration prepare-symbol "$SYM" "$TF" \
  --download --skip-final \
  --config config/profiles/canonical_4model.yaml \
  --max-trials 10
```

### WSL — быстрее по шагам

Вместо одной `prepare-symbol` используйте шаги 2–5 ниже (контроль на каждом этапе).

### Проверка

```bash
ls -lh data/ohlcv/BTC-USDT_1h.parquet data/features/BTC-USDT_1h.parquet
test -f artifacts/BTC-USDT_1h/manifest.json && echo bundle OK
```

### GUI после WSL

**Задачи** → Pipeline: чеклист ohlcv / features / bundle; Toolbar **Обновить**.

---

## Шаг 2 — Признаки (`build-features`)

### GUI (эквивалент кнопки)

| Действие | Ожидание |
|----------|----------|
| **Задачи** → **Pipeline** | |
| **Признаки** | Лог CLI: exit 0; Data health: `features_rows` > 0 |

### WSL

```bash
cd /mnt/d/IST && source .venv/bin/activate
export SYM="BTC/USDT" TF="1h"

python -m orchestration build-features --symbol "$SYM" --timeframe "$TF" \
  --config config/profiles/canonical_4model.yaml
```

Пересборка принудительно:

```bash
python -m orchestration build-features --symbol "$SYM" --timeframe "$TF" --force
```

### Проверка

```bash
ls -lh data/features/BTC-USDT_1h.parquet
python -m orchestration list-symbols | head -80
```

### GUI после WSL

| Куда | Что увидеть |
|------|-------------|
| **Задачи** → Pipeline | Чеклист: features ✓ (может потребоваться **Обновить** на вкладке) |
| **Задачи** → Pipeline | Data health: число строк features |
| **График** → **IST** | Свечи (если уже был OHLCV) |

---

## Шаг 3 — Тюнинг (`tune-thesis`) — долгий; есть быстрый вариант

### GUI

| Действие | Ожидание |
|----------|----------|
| **Задачи** → Pipeline | Фаза: `fast` / `all` / `confirm` |
| **Тюнинг thesis** | Лог CLI (долго); новые строки в журнале |

### WSL — быстрая проверка (рекомендуется сначала)

```bash
cd /mnt/d/IST && source .venv/bin/activate
export SYM="BTC/USDT" TF="1h"

# только fast-фаза, turbo-профиль, мало строк
python -m orchestration tune-thesis \
  --symbol "$SYM" --timeframe "$TF" \
  --phase fast \
  --profile turbo \
  --config config/profiles/canonical_4model.yaml
```

### WSL — полный тюнинг (диплом)

```bash
python -m orchestration tune-thesis \
  --symbol "$SYM" --timeframe "$TF" \
  --phase all \
  --config config/profiles/canonical_4model.yaml
```

Только confirm (после fast/refine):

```bash
python -m orchestration tune-thesis \
  --symbol "$SYM" --timeframe "$TF" \
  --phase confirm
```

### Проверка

```bash
tail -3 docs/backtest_journal/runs.jsonl
grep -o '"acceptance_passed": [^,]*' docs/backtest_journal/runs.jsonl | tail -5
test -f config/symbols/BTC-USDT_1h.yaml && grep -A2 orchestration_tuning_best config/symbols/BTC-USDT_1h.yaml | head -20
```

### GUI после WSL

| Куда | Что увидеть |
|------|-------------|
| **Задачи** → **Журнал WFO** | **Обновить** → новые строки, stage/label |
| **Конфигурация** | Обновились overrides / tuning-best (если confirm записал) |

---

## Шаг 4 — Финальное обучение (`train-final-symbol`)

### GUI

| Действие | Ожидание |
|----------|----------|
| **Задачи** → Pipeline | **Финальное обучение** |
| Toolbar | Появится `run_id` bundle |
| **График** → **Bundle** | **Обновить** — model_keys lgb, gru, xgb, cnn |

### WSL

```bash
cd /mnt/d/IST && source .venv/bin/activate
export SYM="BTC/USDT" TF="1h"

python -m orchestration train-final-symbol \
  --symbol "$SYM" --timeframe "$TF" --force
```

### Проверка

```bash
ls -la artifacts/BTC-USDT_1h/
python -m orchestration manifest-show --symbol BTC/USDT --timeframe 1h | head -60
test -f artifacts/BTC-USDT_1h/manifest.json && echo "manifest OK"
```

### GUI после WSL

| Куда | Что увидеть |
|------|-------------|
| Toolbar | bundle ● зелёный, текст run_id |
| **График** → **Решение** | 4 модели, направление, P(up) |
| **График** → **IST** + ☑ Сигналы | Маркеры на хвосте |
| **График** → **Bundle** | manifest, веса |

---

## Шаг 5 — Отчёт WFO (`report-real`) — главная проверка журнала

### GUI (как кнопка «Отчёт WFO»)

| Элемент | Рекомендация |
|---------|--------------|
| ☑ **Кэш фичей** | вкл |
| ☑ **tuning-best** | вкл |
| **Баров 0=YAML** | `8000` для быстрого теста или `0` для полного хвоста |
| ☐ **Все 4 модели** | вкл для полного TF |

### WSL — быстрый отчёт (5–20 мин)

```bash
cd /mnt/d/IST && source .venv/bin/activate
export SYM="BTC/USDT" TF="1h"

python -m orchestration report-real \
  --symbol "$SYM" \
  --timeframe "$TF" \
  --parquet data/ohlcv/BTC-USDT_1h.parquet \
  --config config.yaml \
  --max-rows 8000 \
  --use-tuning-best \
  --use-feature-cache \
  --dl-epochs 3
```

### WSL — полный acceptance-прогон

```bash
python -m orchestration report-real \
  --symbol "$SYM" \
  --timeframe "$TF" \
  --parquet data/ohlcv/BTC-USDT_1h.parquet \
  --config config.yaml \
  --max-rows 0 \
  --use-tuning-best \
  --use-feature-cache \
  --full-models \
  --dl-epochs 3
```

Запись только в JSON (без журнала — редко нужно):

```bash
python -m orchestration report-real \
  --symbol "$SYM" --timeframe "$TF" \
  --use-tuning-best --use-feature-cache --max-rows 8000 \
  --json-out docs/thesis_report_BTC_quick.json
```

### Проверка

```bash
tail -1 docs/backtest_journal/runs.jsonl | python -m json.tool
ls -t docs/backtest_journal/runs/*.json | head -1
# подставьте RUN_ID из имени файла:
RUN_ID=$(ls -t docs/backtest_journal/runs/*.json | head -1 | xargs basename -s .json)
python -c "import json; d=json.load(open('docs/backtest_journal/runs/${RUN_ID}.json')); print('accept', d.get('acceptance_passed'), 'sharpe', d.get('summary',{}).get('mean_sharpe'))"
```

### GUI после WSL

| Куда | Что увидеть |
|------|-------------|
| Toolbar | **acc:PASS** или **acc:FAIL** |
| **Задачи** → **Журнал WFO** | **Обновить** → новая строка сверху |
| Клик по строке | **Сводка**: folds, Sharpe, PF, WFE, acceptance_checks |
| Низ | График equity по фолдам |
| **Журнал** | **Лучший PASS** — прыжок к лучшему run |

---

## Шаг 6 — Explain (сверка с «Решение»)

### GUI

**График** → подвкладка **Решение** — запомните signal, P(up), блокировку.

### WSL

```bash
cd /mnt/d/IST && source .venv/bin/activate
python -m orchestration explain --symbol BTC/USDT --timeframe 1h --window 256
```

### Проверка

Числа в CLI должны совпасть с карточкой **Решение** (после **Обновить** в GUI).

---

## Шаг 7 — Режим

### GUI

**Режим** → **Загрузить** → таблица, % trend/range.

### WSL

```bash
cd /mnt/d/IST && source .venv/bin/activate
python -m orchestration regime-history --symbol BTC/USDT --timeframe 1h --step 1 | head -40
```

### GUI после WSL

Те же порядки величин % в статистике под таблицей.

---

## Шаг 8 — Paper

### GUI

| Действие | Ожидание |
|----------|----------|
| **Задачи** → **Paper** | |
| **Подключить paper** | Баланс ~10000 USDT |
| Combo **Вручную** | |
| **Выполнить шаг** | Лог + график баланса + сверка |

### WSL

Paper только в GUI (API). Для сравнения сигнала:

```bash
python -m orchestration explain --symbol BTC/USDT --timeframe 1h
```

### GUI

Сверка: если signal=0 и dead-zone — Paper и Backtester оба 0% (нормально).

---

## Быстрый контур «всё обучение за один заход» (WSL)

Скопируйте блок целиком (30–90+ мин в зависимости от GPU и данных):

```bash
cd /mnt/d/IST && source .venv/bin/activate
export TF_CPP_MIN_LOG_LEVEL=2
export SYM="BTC/USDT" TF="1h"

# 0) данные (пропустите если parquet уже есть)
test -f data/ohlcv/BTC-USDT_1h.parquet || \
  python -m data_layer -s BTC/USDT -t 1h --from 2023-01-01

# 1) признаки
python -m orchestration build-features --symbol "$SYM" --timeframe "$TF"

# 2) тюнинг — быстрая фаза (замените на --phase all для диплома)
python -m orchestration tune-thesis --symbol "$SYM" --timeframe "$TF" \
  --phase fast --profile turbo --config config/profiles/canonical_4model.yaml

# 3) финальный bundle
python -m orchestration train-final-symbol --symbol "$SYM" --timeframe "$TF" --force

# 4) WFO отчёт
python -m orchestration report-real \
  --symbol "$SYM" --timeframe "$TF" \
  --parquet data/ohlcv/BTC-USDT_1h.parquet \
  --config config.yaml \
  --max-rows 8000 --use-tuning-best --use-feature-cache --dl-epochs 3

# 5) explain для сверки с GUI
python -m orchestration explain --symbol "$SYM" --timeframe "$TF"

echo "=== done ==="
tail -1 docs/backtest_journal/runs.jsonl
```

Затем в GUI: **BTC/USDT 1h** → **Журнал WFO** → **Обновить** → **График** (IST, Решение) → **Paper** (1 шаг).

---

## Чеклист GUI после WSL-контура

| # | Вкладка | Действие | OK |
|---|---------|----------|-----|
| 1 | Toolbar | acc + bundle run_id | ☐ |
| 2 | **График** OKX | live | ☐ |
| 3 | **График** IST | regime + сигналы | ☐ |
| 4 | **Решение** | 4 модели, согласно `explain` | ☐ |
| 5 | **Режим** | таблица | ☐ |
| 6 | **Bundle** | lgb/gru/xgb/cnn | ☐ |
| 7 | **Журнал** | последний run + сводка | ☐ |
| 8 | **Paper** | шаг + баланс | ☐ |

---

## Мини-скрипт только проверки артефактов (без обучения)

```bash
cd /mnt/d/IST && source .venv/bin/activate
echo "OHLCV:" && ls -lh data/ohlcv/BTC-USDT_1h.parquet 2>&1
echo "Features:" && ls -lh data/features/BTC-USDT_1h.parquet 2>&1
echo "Bundle:" && ls artifacts/BTC-USDT_1h/manifest.json 2>&1
echo "Journal lines:" && wc -l docs/backtest_journal/runs.jsonl
python -m orchestration list-symbols
```

---

## Типичные проблемы

| Симптом | WSL | GUI |
|---------|-----|-----|
| Кнопки Pipeline серые | — | без `--demo` |
| `n_folds=0` | `--max-rows 8000` или больше | — |
| Журнал пустой | сначала `report-real` | **Журнал** → Обновить |
| Решение «нет bundle» | `train-final-symbol --force` | **Bundle** → Обновить |
| GUI не видит новый run | — | Toolbar **Обновить** / переключить вкладку |

---

## История

| Дата | Изменение |
|------|-----------|
| 2026-05-23 | Отдельный файл: GUI + WSL на каждый шаг, быстрый контур обучения |
