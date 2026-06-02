# Configuration

## Канон (диплом / основной pipeline)

| Файл | Назначение |
|------|------------|
| **`profiles/canonical_4model.yaml`** | Полный профиль: 4 модели, MTF, WFO, risk bridge |
| **`../config.yaml`** | Корневой конфиг приложения; секция `orchestration` синхронизирована с canonical |

Загрузка оркестратора:

```python
from orchestration.orchestrator_config import OrchestratorConfig
cfg = OrchestratorConfig.from_yaml("config/profiles/canonical_4model.yaml")
# или
cfg = OrchestratorConfig.from_yaml("config.yaml")
```

Валидация:

```bash
python -m orchestration validate-config --config config/profiles/canonical_4model.yaml
```

## Per-symbol overrides

`symbols/BTC-USDT_1h.yaml` — результат **тюнинга** по символу (может отличаться от canonical). Для ВКР фиксируйте, какой профиль использовался в эксперименте.

## Архив (не канон)

`archive/discussion/` — эксперименты для обсуждения (например, только lgb+xgb, `fixed_range`). **Не использовать** как default в CLI.

## Документация для ВКР

Сводка «как реализовано в коде»: [docs/vkr/README.md](../docs/vkr/README.md).

## План работ

См. [docs/IMPLEMENTATION_PLAN.md](../docs/IMPLEMENTATION_PLAN.md).
