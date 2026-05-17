"""
Сохранение / загрузка артефактов оркестратора (единый bundle).

manifest.json + pickle файлы моделей и regime detector. Схема фич —
``feature_columns`` + хеш в manifest (для сверки с ``ModelRegistry.validate_schema``).

Стандартный сценарий: ``save_orchestrator_bundle`` после обучения →
``load_orchestrator_bundle`` → ``InferenceOrchestrator.initialize(...)``.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hashlib

from orchestration.orchestrator_config import OrchestratorConfig


def _feature_schema_hash(columns: List[str]) -> str:
    h = hashlib.sha256(",".join(sorted(columns)).encode()).hexdigest()
    return h[:16]


@dataclass
class OrchestratorBundleManifest:
    bundle_version: str
    created_at: str
    orchestrator_config: Dict[str, Any]
    feature_columns: List[str]
    feature_schema_hash: str
    model_paths: Dict[str, str]
    regime_detector_path: str
    train_meta_threshold: Optional[float] = None


def save_orchestrator_bundle(
    bundle_dir: str | Path,
    config: OrchestratorConfig,
    models: Dict[str, Any],
    regime_detector: Any,
    feature_columns: List[str],
    train_meta_threshold: Optional[float] = None,
) -> Path:
    """
    Сохраняет manifest и pickle для каждой модели + regime detector.
    """
    root = Path(bundle_dir)
    art = root / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    model_paths: Dict[str, str] = {}
    for k, m in models.items():
        rel = f"artifacts/{k}.pkl"
        with open(root / rel, "wb") as f:
            pickle.dump(m, f, protocol=pickle.HIGHEST_PROTOCOL)
        model_paths[k] = rel
    rd_rel = "artifacts/regime_detector.pkl"
    with open(root / rd_rel, "wb") as f:
        pickle.dump(regime_detector, f, protocol=pickle.HIGHEST_PROTOCOL)
    man = OrchestratorBundleManifest(
        bundle_version="1",
        created_at=datetime.now(timezone.utc).isoformat(),
        orchestrator_config=config.to_dict(),
        feature_columns=list(feature_columns),
        feature_schema_hash=_feature_schema_hash(feature_columns),
        model_paths=model_paths,
        regime_detector_path=rd_rel,
        train_meta_threshold=train_meta_threshold,
    )
    d = asdict(man)
    (root / "manifest.json").write_text(json.dumps(d, indent=2), encoding="utf-8")
    return root / "manifest.json"


def load_orchestrator_bundle(
    bundle_dir: str | Path,
) -> Tuple[OrchestratorConfig, Dict[str, Any], Any, List[str], Optional[float], str]:
    """
    Returns:
        config, models dict, regime_detector, feature_columns, train_meta_threshold, schema_hash
    """
    root = Path(bundle_dir)
    raw = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    cfg = OrchestratorConfig.from_dict(raw["orchestrator_config"])
    models: Dict[str, Any] = {}
    for k, rel in raw["model_paths"].items():
        with open(root / rel, "rb") as f:
            models[k] = pickle.load(f)
    with open(root / raw["regime_detector_path"], "rb") as f:
        regime = pickle.load(f)
    feat = list(raw["feature_columns"])
    thr = raw.get("train_meta_threshold")
    sh = raw.get("feature_schema_hash", "")
    return cfg, models, regime, feat, thr, sh


def validate_bundle_feature_schema(bundle_dir: str | Path, current_columns: List[str]) -> bool:
    """True если хеш списка фич совпадает с manifest."""
    root = Path(bundle_dir)
    raw = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    expected = raw.get("feature_schema_hash")
    if not expected:
        return False
    return expected == _feature_schema_hash(current_columns)
