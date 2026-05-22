"""Build orchestration CLI argument lists for GUI subprocess runner."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

from orchestration.symbols import REPO_ROOT, paths_for, tuning_best_for


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

    def build_features_cmd(
        self,
        symbol: str,
        timeframe: str = "1h",
        *,
        config: str = "config/profiles/canonical_4model.yaml",
        force: bool = False,
    ) -> List[str]:
        args = [
            self.python_exe(),
            "-m",
            "orchestration",
            "build-features",
            "--symbol",
            symbol,
            "--timeframe",
            timeframe,
            "--config",
            config,
        ]
        if force:
            args.append("--force")
        return args

    def tune_thesis_cmd(
        self,
        symbol: str,
        timeframe: str = "1h",
        *,
        phase: str = "all",
        config: str = "config/profiles/thesis_tuning.yaml",
    ) -> List[str]:
        return [
            self.python_exe(),
            "-m",
            "orchestration",
            "tune-thesis",
            "--symbol",
            symbol,
            "--timeframe",
            timeframe,
            "--phase",
            phase,
            "--config",
            config,
        ]

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
        max_rows: int = 0,
        config: str = "config.yaml",
        full_models: bool = False,
        use_tuning_best: bool = True,
        use_feature_cache: bool = True,
        json_out: Optional[str] = None,
    ) -> List[str]:
        sp = paths_for(symbol, timeframe)
        args = [
            self.python_exe(),
            "-m",
            "orchestration",
            "report-real",
            "--symbol",
            symbol,
            "--parquet",
            str(sp.parquet),
            "--config",
            config,
            "--max-rows",
            str(max_rows),
        ]
        if use_tuning_best:
            args.append("--use-tuning-best")
        if use_feature_cache:
            args.append("--use-feature-cache")
        if full_models:
            args.append("--full-models")
        if json_out:
            args.extend(["--json-out", json_out])
        return args

    def default_report_max_rows(self, symbol: str, timeframe: str = "1h") -> int:
        tb = tuning_best_for(symbol, timeframe)
        return int(tb.get("max_rows") or 8000)
