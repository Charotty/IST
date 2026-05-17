"""Artifact bundle introspection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from orchestration.artifact_bundle import load_orchestrator_bundle, validate_bundle_feature_schema
from orchestration.symbols import paths_for

from gui.api.types import BundleInfo


class BundlesApi:
    def latest_bundle_dir(self, symbol: str, timeframe: str = "1h") -> Optional[Path]:
        sp = paths_for(symbol, timeframe)
        return sp.latest_bundle()

    def bundle_info(
        self,
        symbol: str,
        timeframe: str = "1h",
        *,
        bundle_dir: Optional[Path] = None,
        validate_schema: bool = True,
    ) -> BundleInfo:
        root = bundle_dir or self.latest_bundle_dir(symbol, timeframe)
        if root is None:
            raise FileNotFoundError(
                f"No artifact bundle for {paths_for(symbol, timeframe).slug}; "
                "run prepare-symbol / train-final first."
            )
        root = Path(root)
        raw = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        cfg, _models, _regime, feat_cols, train_thr, schema_hash = load_orchestrator_bundle(root)
        schema_valid = None
        if validate_schema:
            try:
                from orchestration.symbol_pipeline import build_features

                feat = build_features(paths_for(symbol, timeframe).parquet)
                schema_valid = validate_bundle_feature_schema(root, list(feat.columns))
            except Exception:
                schema_valid = False

        return BundleInfo(
            bundle_dir=root,
            run_id=root.name,
            created_at=raw.get("created_at", ""),
            feature_columns=list(feat_cols),
            feature_schema_hash=schema_hash or raw.get("feature_schema_hash", ""),
            train_meta_threshold=train_thr,
            model_keys=list(cfg.model_keys),
            orchestrator_config=cfg.to_dict(),
            schema_valid=schema_valid,
        )
