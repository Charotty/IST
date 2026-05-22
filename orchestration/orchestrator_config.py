"""
Orchestrator Configuration

Fixed configuration for unified orchestration.
Defines model keys, regime keys, and pipeline parameters.
"""

from dataclasses import dataclass, field, asdict, fields
from typing import Dict, List, Optional
from pathlib import Path
import warnings
import yaml


def _renormalize_weight_subset(template: Dict[str, float], model_keys: List[str]) -> Dict[str, float]:
    """Restrict default weight templates to the active model_keys and renormalize to sum 1."""
    picked = {k: template[k] for k in model_keys if k in template}
    if not picked:
        n = len(model_keys)
        return {k: 1.0 / n for k in model_keys}
    total = sum(picked.values())
    return {k: v / total for k, v in picked.items()}


_DEFAULT_TREND_WEIGHTS = {'lgb': 0.10, 'gru': 0.45, 'xgb': 0.10, 'cnn': 0.35}
_DEFAULT_RANGE_WEIGHTS = {'lgb': 0.55, 'gru': 0.10, 'xgb': 0.25, 'cnn': 0.10}
_DEFAULT_BREAKOUT_WEIGHTS = {'lgb': 0.25, 'gru': 0.25, 'xgb': 0.25, 'cnn': 0.25}
@dataclass
class OrchestratorConfig:
    """
    Unified orchestrator configuration with fixed model keys.
    
    Pipeline Contract:
    features[t] → regime[t] → {p_lgb, p_gru, p_xgb, p_cnn, ...}[t] → meta_mgmt[t] → direction_soft[t] → decision → risk
    """
    
    # Fixed model keys - MUST match actual model implementations
    model_keys: List[str] = field(default_factory=lambda: ['lgb', 'gru', 'xgb', 'cnn'])
    
    # Regime keys for regime detection
    regime_keys: List[str] = field(default_factory=lambda: ['trend', 'range', 'breakout'])
    
    # Thresholds
    direction_threshold: float = 0.52
    meta_threshold: float = 0.5
    signal_threshold: float = 0.6  # Legacy single-step direction conversion when decision pipeline is off
    meta_threshold_mode: str = "median"  # median | mean | fixed (for train-only calibration)
    threshold_rolling_window: int = 100  # Causal rolling meta threshold (decision + safe path)
    apply_decision_pipeline: bool = True  # Use DecisionPipeline (integrated) for final signals
    trade_mode: str = "both"  # both | long_only | short_only
    min_signal_margin: float = 0.0  # |p-0.5| must exceed this to keep a directional leg
    signal_strategy: str = "ensemble"  # ensemble | momentum_confirm
    momentum_sma_period: int = 100
    label_min_return: float = 0.0  # min forward return for positive class (0 = any up move)
    volatility_filter_percentile: float = 0.0  # 0=off; e.g. 90 → flat when ATR > train p90
    max_position_fraction: float = 1.0  # cap notional/account per bar (ATR sizing)

    # Ensemble mode
    ensemble_mode: str = "regime_adaptive"  # 'regime_adaptive', 'fixed_trend', 'fixed_range'
    
    # Regime-specific weights (must match model_keys)
    trend_weights: Dict[str, float] = field(default_factory=dict)
    range_weights: Dict[str, float] = field(default_factory=dict)
    breakout_weights: Dict[str, float] = field(default_factory=dict)
    
    # Training parameters
    train_window_size: int = 1000
    test_window_size: int = 200
    walk_forward_step: int = 100
    max_wfo_folds: int = 0  # 0 = all folds; >0 caps WFO folds (faster tuning)

    # Feature parameters
    feature_window_size: int = 24
    
    # Data leakage prevention
    prediction_horizon: int = 12  # Prediction horizon in bars
    embargo_period: int = 5  # Embargo period between train and test in bars
    enable_purge: bool = True  # Whether to apply purge (remove H bars from end of train)
    allow_meta_label_on_test: bool = False  # Never allow meta-label on test
    safe_label_generation: bool = True  # Use safe label generation by default

    # Profile YAML fields (tuning / risk bridge; not all used inside WFO loop)
    use_risk_bridge: bool = False
    safe_threshold_mode: bool = True
    apply_atr_trailing: bool = False

    def __post_init__(self):
        """Initialize default weights if not provided."""
        # Default trend weights: sequential models dominant
        if not self.trend_weights:
            self.trend_weights = _renormalize_weight_subset(_DEFAULT_TREND_WEIGHTS, self.model_keys)
        
        # Default range weights: tabular models dominant
        if not self.range_weights:
            self.range_weights = _renormalize_weight_subset(_DEFAULT_RANGE_WEIGHTS, self.model_keys)
        
        # Default breakout weights: balanced
        if not self.breakout_weights:
            self.breakout_weights = _renormalize_weight_subset(_DEFAULT_BREAKOUT_WEIGHTS, self.model_keys)
        
        # Validate weights match model_keys
        self._validate_weights()
    
    def _validate_weights(self):
        """Validate that weight keys match model_keys."""
        for regime_name, weights in [('trend', self.trend_weights), ('range', self.range_weights), ('breakout', self.breakout_weights)]:
            weight_keys = set(weights.keys())
            model_key_set = set(self.model_keys)
            
            if weight_keys != model_key_set:
                raise ValueError(
                    f"{regime_name}_weights keys {weight_keys} do not match model_keys {model_key_set}. "
                    f"Ensure all models in model_keys have weights defined."
                )
    
    def get_regime_weights(self, regime: str) -> Dict[str, float]:
        """
        Get weights for a specific regime.
        
        Args:
            regime: Regime name ('trend', 'range', or 'breakout')
            
        Returns:
            Dictionary of model weights
        """
        if regime == 'trend':
            return self.trend_weights
        elif regime == 'range':
            return self.range_weights
        elif regime == 'breakout':
            return self.breakout_weights
        else:
            raise ValueError(f"Unknown regime: {regime}. Must be one of {self.regime_keys}")
    
    def set_regime_weights(self, regime: str, weights: Dict[str, float]):
        """
        Set weights for a specific regime.
        
        Args:
            regime: Regime name ('trend', 'range', or 'breakout')
            weights: Dictionary of model weights (must match model_keys)
        """
        # Validate keys
        if set(weights.keys()) != set(self.model_keys):
            raise ValueError(
                f"Weight keys {set(weights.keys())} must match model_keys {set(self.model_keys)}"
            )
        
        # Normalize weights
        total = sum(weights.values())
        normalized_weights = {k: v / total for k, v in weights.items()}
        
        if regime == 'trend':
            self.trend_weights = normalized_weights
        elif regime == 'range':
            self.range_weights = normalized_weights
        elif regime == 'breakout':
            self.breakout_weights = normalized_weights
        else:
            raise ValueError(f"Unknown regime: {regime}. Must be one of {self.regime_keys}")
    
    @classmethod
    def from_yaml(cls, yaml_path: Path) -> 'OrchestratorConfig':
        """
        Load configuration from YAML file.
        
        Unknown keys in the orchestration section are ignored (with a warning) so the
        project ``config.yaml`` can carry documentation-only or future-facing fields.
        
        Args:
            yaml_path: Path to YAML configuration file
            
        Returns:
            OrchestratorConfig instance
        """
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Extract orchestration section if present
        if config_dict is None:
            config_dict = {}
        elif 'orchestration' in config_dict:
            config_dict = config_dict['orchestration'] or {}
        
        allowed = {f.name for f in fields(cls)}
        unknown = set(config_dict) - allowed
        if unknown:
            warnings.warn(
                f"Ignoring unknown orchestration YAML keys: {sorted(unknown)}",
                UserWarning,
                stacklevel=2,
            )
        filtered = {k: config_dict[k] for k in config_dict if k in allowed}
        return cls(**filtered)
    
    def to_yaml(self, yaml_path: Path):
        """
        Save configuration to YAML file.
        
        Args:
            yaml_path: Path to save YAML configuration file
        """
        config_dict = asdict(self)
        
        with open(yaml_path, 'w') as f:
            yaml.dump({'orchestration': config_dict}, f, default_flow_style=False)
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'OrchestratorConfig':
        """
        Load configuration from dictionary.
        
        Unknown keys are ignored with a warning.
        
        Args:
            config_dict: Dictionary with configuration values
            
        Returns:
            OrchestratorConfig instance
        """
        allowed = {f.name for f in fields(cls)}
        unknown = set(config_dict) - allowed
        if unknown:
            warnings.warn(
                f"Ignoring unknown orchestration dict keys: {sorted(unknown)}",
                UserWarning,
                stacklevel=2,
            )
        filtered = {k: config_dict[k] for k in config_dict if k in allowed}
        return cls(**filtered)
