# Intelligent Trading System

ITS is a Python research and engineering pipeline for algorithmic crypto trading.

The project is now organized around one core flow:

```text
data_layer -> preprocessing -> storage -> features -> targets -> models -> metalearning -> decision -> backtesting/execution -> monitoring/mlops
```

UI, dashboard, Telegram, CLI demo wrappers, and generated experiment artifacts were removed from the core tree so the project can be evaluated as a pipeline-first codebase.

## Quick Checks

Use the bundled virtual environment from the repository root:

```powershell
$env:PYTHONPATH='D:\IST'
.venv\Scripts\python.exe its_project\run.py check
.venv\Scripts\python.exe its_project\run.py demo --samples 300 --estimators 10
.venv\Scripts\python.exe -m pytest its_project\tests --collect-only -q -p no:cacheprovider -p no:cov -o addopts=''
```

## Core Documentation

The layer-by-layer ownership map is in:

```text
its_project/docs/CORE_PIPELINE.md
```

That document explains what each layer contains, what should stay, and what still needs review.

## Main Entry Point

`its_project/run.py` is the single launcher for the refactored core:

- `check`: validates the expected core layer structure.
- `layers`: prints the canonical layer map.
- `demo`: runs a synthetic end-to-end ML pipeline.
- `test --collect-only`: verifies pytest import/discovery health.

## Current Status

This is still a research-grade trading system, not a production trading bot.

What improved in this refactor:

- Removed UI/interface sprawl from the core scope.
- Restored missing `preprocessing` and `config` modules expected by tests.
- Added `features.basic`.
- Made optional-heavy package imports lazier.
- Fixed test collection blockers.
- Replaced the old multi-launcher with a single pipeline launcher.

Next engineering quality gates:

- Run and fix the full test suite, not only collection.
- Consolidate duplicate risk/sizing/order-manager APIs.
- Fix `data_layer/reconnection_manager.py` before making it part of eager imports.
- Decide whether `backend`, `evaluation`, `training`, and report scripts stay as separate optional packages or move out of the core repository.
