from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from its_project.features.feature_config import FeatureConfig
from its_project.features.preprocessing import normalize_features
from its_project.features.technical import TechnicalFeatures
from its_project.features.orderbook import OrderBookFeatures
from its_project.features.microstructure import MicrostructureFeatures

logger = logging.getLogger(__name__)


class FeatureBuilder:
    """Builds and manages features with configuration-driven selection."""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        self._feature_calculators = {}
        self._feature_names: List[str] = []
        self._scaling_stats: Optional[Dict] = None
        
    def _initialize_calculators(self) -> None:
        """Initialize feature calculators based on config."""
        active_features = self.config.get_active_features()
        
        if active_features["technical"]:
            self._feature_calculators["technical"] = TechnicalFeatures({
                "indicators": active_features["technical"]
            })
        
        if active_features["lob"]:
            self._feature_calculators["lob"] = OrderBookFeatures({
                "depth_levels": len([f for f in active_features["lob"] if f.startswith("ofi_")]),
                "imbalance_levels": 10
            })
        
        if active_features["microstructure"]:
            self._feature_calculators["microstructure"] = MicrostructureFeatures({
                "window": 20
            })
    
    def calculate_features(self, data: pd.DataFrame, fit_scaling: bool = True) -> np.ndarray:
        """
        Calculate features from synchronized data.
        
        Args:
            data: Synchronized DataFrame with OHLCV and orderbook data
            fit_scaling: Whether to fit scaling parameters (use False for inference)
            
        Returns:
            Feature matrix with shape (n_samples, n_features)
        """
        if not self._feature_calculators:
            self._initialize_calculators()
        
        # Calculate features from each calculator
        feature_matrices = []
        feature_names = []
        
        for name, calculator in self._feature_calculators.items():
            try:
                features = calculator.calculate(data)
                names = calculator.get_feature_names()
                
                # Filter excluded features
                active_names = self.config.validate_features(names)
                if active_names:
                    # Select only active columns
                    active_indices = [i for i, n in enumerate(names) if n in active_names]
                    active_features = features[:, active_indices]
                    
                    feature_matrices.append(active_features)
                    feature_names.extend(active_names)
                    
            except Exception as e:
                logger.warning(f"Failed to calculate {name} features: {e}")
                continue
        
        if not feature_matrices:
            raise ValueError("No features could be calculated")
        
        # Concatenate all features
        feature_matrix = np.hstack(feature_matrices)
        self._feature_names = feature_names
        
        # Apply scaling
        if self.config.scaling_method == "zscore":
            if fit_scaling:
                normalized, stats = normalize_features(
                    feature_matrix, method="zscore", axis=0
                )
                self._scaling_stats = stats
            else:
                if self._scaling_stats is None:
                    raise ValueError("Scaling stats not available. Call with fit_scaling=True first.")
                normalized = self._apply_scaling(feature_matrix)
        else:
            normalized = feature_matrix
        
        # Validate features
        normalized = self._validate_features(normalized)
        
        return normalized
    
    def _apply_scaling(self, features: np.ndarray) -> np.ndarray:
        """Apply previously fitted scaling."""
        if self._scaling_stats is None:
            return features
        
        if self.config.scaling_method == "zscore":
            mean = self._scaling_stats["mean"]
            std = self._scaling_stats["std"]
            return (features - mean) / (std + 1e-10)
        
        return features
    
    def _validate_features(self, features: np.ndarray) -> np.ndarray:
        """Validate and clean features."""
        # Check for constant features
        std_per_feature = np.std(features, axis=0)
        constant_mask = std_per_feature < self.config.min_feature_std
        
        if np.any(constant_mask):
            logger.warning(f"Removing {np.sum(constant_mask)} constant features")
            features = features[:, ~constant_mask]
            self._feature_names = [n for i, n in enumerate(self._feature_names) if not constant_mask[i]]
        
        # Handle NaN values
        nan_ratio = np.isnan(features).sum(axis=0) / features.shape[0]
        nan_mask = nan_ratio > self.config.max_nan_ratio
        
        if np.any(nan_mask):
            logger.warning(f"Removing {np.sum(nan_mask)} features with too many NaNs")
            features = features[:, ~nan_mask]
            self._feature_names = [n for i, n in enumerate(self._feature_names) if not nan_mask[i]]
        
        # Fill remaining NaNs
        features = np.nan_to_num(features, nan=0.0)
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get names of calculated features."""
        return self._feature_names.copy()
    
    def get_scaling_stats(self) -> Optional[Dict]:
        """Get scaling statistics for reproducibility."""
        return self._scaling_stats.copy() if self._scaling_stats else None
    
    def set_scaling_stats(self, stats: Dict) -> None:
        """Set scaling statistics (e.g., loaded from disk)."""
        self._scaling_stats = stats.copy()
