"""Active pipeline task logs (``artifacts/<slug>/active/*.jsonl``)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from orchestration.symbols import paths_for

from gui.api.types import JobEvent, PipelineStepStatus


class JobsApi:
    def list_active_tasks(self, symbol: str, timeframe: str = "1h") -> List[str]:
        active = paths_for(symbol, timeframe).artifacts_root / "active"
        if not active.is_dir():
            return []
        return sorted(p.stem for p in active.glob("*.jsonl"))

    def read_task_log(
        self,
        symbol: str,
        timeframe: str,
        task_id: str,
        *,
        tail: int = 200,
    ) -> List[JobEvent]:
        path = paths_for(symbol, timeframe).artifacts_root / "active" / f"{task_id}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        lines = path.read_text(encoding="utf-8").splitlines()
        if tail > 0:
            lines = lines[-tail:]
        events: List[JobEvent] = []
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                raw = {"message": line}
            events.append(JobEvent(task_id=task_id, line_no=i + 1, raw=raw))
        return events

    def pipeline_status(self, symbol: str, timeframe: str = "1h") -> List[PipelineStepStatus]:
        """Heuristic stage checklist from filesystem (no running subprocess)."""
        sp = paths_for(symbol, timeframe)
        from orchestration.canonical_pipeline import features_parquet_for

        feat = features_parquet_for(sp.symbol, sp.timeframe)
        bundle = sp.latest_bundle()
        return [
            PipelineStepStatus(
                "ohlcv",
                sp.parquet.is_file(),
                str(sp.parquet) if sp.parquet.is_file() else "missing",
            ),
            PipelineStepStatus(
                "features",
                feat.is_file(),
                str(feat) if feat.is_file() else "missing",
            ),
            PipelineStepStatus(
                "symbol_config",
                sp.config_yaml.is_file(),
                str(sp.config_yaml) if sp.config_yaml.is_file() else "optional",
            ),
            PipelineStepStatus(
                "artifact_bundle",
                bundle is not None,
                bundle.name if bundle else "no LATEST",
            ),
        ]
