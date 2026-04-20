from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    
    # Technical indicators to include
    technical_indicators: List[str] = field(default_factory=lambda: [
        "rsi", "macd", "bbands", "atr", "stoch"
    ])
    
    # Order book features to include
    lob_features: List[str] = field(default_factory=lambda: [
        "spread", "spread_pct", "volume_imbalance", 
        "bid_depth", "ask_depth", "bid_vwap", "ask_vwap",
        "bid_density", "ask_density", "ofi_0", "ofi_1", "ofi_2", "ofi_3", "ofi_4"
    ])
    
    # Microstructure features to include
    microstructure_features: List[str] = field(default_factory=lambda: [
        "roll_impact", "vpin", "realized_volatility", "amihud_illiquidity", "kyle_lambda"
    ])
    
    # Features to exclude (noisy or redundant)
    exclude_features: List[str] = field(default_factory=lambda: [
        # Potentially noisy features
        "ofi_5", "ofi_6", "ofi_7", "ofi_8", "ofi_9",  # Deep OFI levels may be noisy
        "bid_density", "ask_density",  # Density measures can be unstable
    ])
    
    # Feature scaling method
    scaling_method: str = "zscore"  # "zscore" or "minmax"
    
    # Scaling parameters
    scaling_window: Optional[int] = None  # If None, use full history
    scaling_rolling: bool = True  # Use rolling window for scaling to avoid leakage
    
    # Feature validation
    min_feature_std: float = 1e-6  # Minimum standard deviation to keep feature
    max_nan_ratio: float = 0.1  # Maximum ratio of NaN values allowed
    
    def get_active_features(self) -> Dict[str, List[str]]:
        """Get active features by category after exclusions."""
        active = {
            "technical": [f for f in self.technical_indicators if f not in self.exclude_features],
            "lob": [f for f in self.lob_features if f not in self.exclude_features],
            "microstructure": [f for f in self.microstructure_features if f not in self.exclude_features],
        }
        return active
    
    def validate_features(self, feature_names: List[str]) -> List[str]:
        """Validate feature names against configuration."""
        active_features = []
        for name in feature_names:
            if name not in self.exclude_features:
                active_features.append(name)
        return active_features
