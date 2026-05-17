"""Build orchestration CLI argument lists for GUI subprocess runner."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

from orchestration.symbols import REPO_ROOT, paths_for


class CliApi:
    """Command builders only — execution is in ``gui.app.workers.CliProcessWorker``."""

    @staticmethod
    def python_exe() -> str:
        return sys.executable

    @staticmethod
    def repo_cwd() -> Path:
        return REPO_ROOT

    def prepare_symbol_cmd(
        self,
        symbol: str,
        timeframe: str = "1h",
        *,
        download: bool = False,
        config: str = "config/profiles/canonical_4model.yaml",
        max_trials: int = 30,
        skip_final: bool = False,
    ) -> List[str]:
        args = [
            self.python_exe(),
            "-m",
            "orchestration",
            "prepare-symbol",
            symbol,
            timeframe,
            "--config",
            config,
            "--max-trials",
            str(max_trials),
        ]
        if download:
            args.append("--download")
        if skip_final:
            args.append("--skip-final")
        return args

    def train_final_cmd(
        self,
        symbol: str,
        timeframe: str = "1h",
        *,
        force: bool = False,
    ) -> List[str]:
        args = [
            self.python_exe(),
            "-m",
            "orchestration",
            "train-final-symbol",
            "--symbol",
            symbol,
            "--timeframe",
            timeframe,
        ]
        if force:
            args.append("--force")
        return args

    def report_real_cmd(
        self,
        symbol: str,
        timeframe: str = "1h",
        *,
        max_rows: int = 8000,
        config: str = "config.yaml",
        full_models: bool = False,
    ) -> List[str]:
        sp = paths_for(symbol, timeframe)
        args = [
            self.python_exe(),
            "-m",
            "orchestration",
            "report-real",
            "--parquet",
            str(sp.parquet),
            "--config",
            config,
            "--max-rows",
            str(max_rows),
        ]
        if full_models:
            args.append("--full-models")
        return args
