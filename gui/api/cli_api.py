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
        config: str = "config/profiles/canonical_4model.yaml",
        tuning_yaml: Optional[str] = None,
    ) -> List[str]:
        args = [
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
        if tuning_yaml:
            args.extend(["--tuning-yaml", tuning_yaml])
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

    def smoke_cmd(self) -> List[str]:
        return [self.python_exe(), "-m", "orchestration", "smoke"]

    def validate_config_cmd(
        self,
        config: str = "config/profiles/canonical_4model.yaml",
    ) -> List[str]:
        return [
            self.python_exe(),
            "-m",
            "orchestration",
            "validate-config",
            "--config",
            config,
        ]

    def tune_until_cmd(
        self,
        symbol: str,
        timeframe: str = "1h",
        *,
        max_rows: int = 8000,
        max_trials: int = 80,
        mode: str = "refine",
        config: str = "config.yaml",
    ) -> List[str]:
        sp = paths_for(symbol, timeframe)
        return [
            self.python_exe(),
            "-m",
            "orchestration",
            "tune-until",
            "--parquet",
            str(sp.parquet),
            "--config",
            config,
            "--max-rows",
            str(max_rows),
            "--max-trials",
            str(max_trials),
            "--mode",
            mode,
        ]

    def from_parquet_cmd(
        self,
        parquet_path: str,
        *,
        config: str = "config/profiles/canonical_4model.yaml",
        full_models: bool = True,
        dl_epochs: int = 3,
        json_out: Optional[str] = None,
        raw_ohlcv: bool = False,
    ) -> List[str]:
        args = [
            self.python_exe(),
            "-m",
            "orchestration",
            "from-parquet",
            parquet_path,
            "--config",
            config,
            "--dl-epochs",
            str(dl_epochs),
        ]
        if full_models:
            args.append("--full-models")
        if raw_ohlcv:
            args.append("--raw-ohlcv")
        if json_out:
            args.extend(["--json-out", json_out])
        return args

    def regime_history_cmd(
        self,
        symbol: str,
        timeframe: str = "1h",
        *,
        step: int = 24,
        json_out: Optional[str] = None,
    ) -> List[str]:
        args = [
            self.python_exe(),
            "-m",
            "orchestration",
            "regime-history",
            "--symbol",
            symbol,
            "--timeframe",
            timeframe,
            "--step",
            str(step),
        ]
        if json_out:
            args.extend(["--json-out", json_out])
        return args

    def list_symbols_cmd(self) -> List[str]:
        return [self.python_exe(), "-m", "orchestration", "list-symbols"]

    def manifest_show_cmd(self, symbol: str, timeframe: str = "1h") -> List[str]:
        return [
            self.python_exe(),
            "-m",
            "orchestration",
            "manifest-show",
            "--symbol",
            symbol,
            "--timeframe",
            timeframe,
        ]

    def features_parquet_path(self, symbol: str, timeframe: str = "1h") -> str:
        from orchestration.canonical_pipeline import features_parquet_for

        sp = paths_for(symbol, timeframe)
        return str(features_parquet_for(sp.symbol, sp.timeframe))
