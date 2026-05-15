"""
Meta Config

Configuration management for meta-learning layer.

Current Configuration:
meta_learning:
  direction_threshold: 0.52
  meta_threshold_mode: "median"
  ensemble:
    default: "regime_adaptive"
    regime_weights:
      trend:
        lgb: 0.10
        lstm: 0.45
        cnn: 0.10
        trans: 0.35
      range:
        lgb: 0.55
        lstm: 0.10
        cnn: 0.25
        trans: 0.10
  dl_window_size: 24
"""

import yaml
from typing import Dict, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class MetaConfig:
    """Configuration for meta-learning layer."""
    
    # Thresholds
    direction_threshold: float = 0.52
    meta_threshold_mode: str = "median"
    default_meta_threshold: float = 0.5
    
    # Ensemble
    ensemble_mode: str = "regime_adaptive"
    
    # Regime weights
    trend_weights: Dict[str, float] = None
    range_weights: Dict[str, float] = None
    
    # Deep learning
    dl_window_size: int = 24
    
    # Calibration
    calibration_method: str = "isotonic"
    
    def __post_init__(self):
        """Set default weights if not provided."""
        if self.trend_weights is None:
            self.trend_weights = {
                'lgb': 0.10,
                'lstm': 0.45,
                'cnn': 0.10,
                'trans': 0.35
            }
        
        if self.range_weights is None:
            self.range_weights = {
                'lgb': 0.55,
                'lstm': 0.10,
                'cnn': 0.25,
                'trans': 0.10
            }
    
    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> 'MetaConfig':
        """
        Load configuration from YAML file.
        
        Args:
            yaml_path: Path to YAML configuration file
            
        Returns:
            MetaConfig instance
        """
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Extract meta_learning section if present
        if 'meta_learning' in config_dict:
            config_dict = config_dict['meta_learning']
        
        return cls(**config_dict)
    
    def to_yaml(self, yaml_path: Union[str, Path]):
        """
        Save configuration to YAML file.
        
        Args:
            yaml_path: Path to save YAML configuration file
        """
        config_dict = asdict(self)
        
        with open(yaml_path, 'w') as f:
            yaml.dump({'meta_learning': config_dict}, f, default_flow_style=False)
    
    def to_dict(self) -> Dict[str, Union[float, str, int, Dict]]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of configuration
        """
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Union[float, str, int, Dict]]) -> 'MetaConfig':
        """
        Load configuration from dictionary.
        
        Args:
            config_dict: Dictionary with configuration values
            
        Returns:
            MetaConfig instance
        """
        return cls(**config_dict)
    
    def update(self, **kwargs):
        """
        Update configuration with new values.
        
        Args:
            **kwargs: Configuration key-value pairs to update
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def get_regime_weights(self, regime: str) -> Dict[str, float]:
        """
        Get weights for a specific regime.
        
        Args:
            regime: Regime name ('trend' or 'range')
            
        Returns:
            Dictionary of model weights
        """
        if regime == 'trend':
            return self.trend_weights
        elif regime == 'range':
            return self.range_weights
        else:
            raise ValueError(f"Unknown regime: {regime}")
    
    def set_regime_weights(self, regime: str, weights: Dict[str, float]):
        """
        Set weights for a specific regime.
        
        Args:
            regime: Regime name ('trend' or 'range')
            weights: Dictionary of model weights
        """
        # Normalize weights
        total = sum(weights.values())
        normalized_weights = {k: v / total for k, v in weights.items()}
        
        if regime == 'trend':
            self.trend_weights = normalized_weights
        elif regime == 'range':
            self.range_weights = normalized_weights
        else:
            raise ValueError(f"Unknown regime: {regime}")
