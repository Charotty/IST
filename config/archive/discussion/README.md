# Архив конфигураций (обсуждение / эксперименты)

Здесь лежат **не канонические** профили. Они не описывают целевую архитектуру диплома (4 модели + regime-adaptive ensemble).

Используйте для:

- сравнения с сокращённым ансамблем (2 модели);
- истории тюнинга Sharpe;
- обсуждения на защите («альтернативный эксперимент»).

**Канон системы:** `config/profiles/canonical_4model.yaml` и секция `orchestration` в корневом `config.yaml`.

| Файл | Содержание |
|------|------------|
| `orchestration_tuning_best_lgb_xgb.yaml` | Только lgb+xgb, `fixed_range`, другие пороги; бывший `orchestration_tuning_best` в root config |
