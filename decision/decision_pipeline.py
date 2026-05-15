"""
Decision Pipeline for ITS

Provides DecisionPipeline class to select and apply signal rules based on configuration.
Serves as the entry point for risk management and backtesting.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Union, Literal
import yaml

from .signal_rules import compute_final_signal, compute_integrated_signal, apply_asymmetric_thresholds


class DecisionPipeline:
    """
    Decision pipeline for signal generation.
    
    Selects between Variant A (final_signal) and Variant B (integrated_signal)
    based on configuration and applies the appropriate rules.
    
    Configuration:
        signal_source: "final" or "integrated"
        direction_threshold: 0.52
        meta_threshold_mode: "median"
        meta_threshold_value: optional fixed threshold
        use_asymmetric_thresholds: False (roadmap feature)
        long_threshold: 0.52 (for asymmetric)
        short_threshold: 0.52 (for asymmetric)
    """
    
    def __init__(self, config: Union[Dict[str, Any], str] = None):
        """
        Initialize decision pipeline.
        
        Args:
            config: Configuration dictionary or path to config file
        """
        if config is None:
            # Default configuration
            self.config = {
                'signal_source': 'integrated',  # recommended
                'direction_threshold': 0.52,
                'meta_threshold_mode': 'median',
                'meta_threshold_value': None,
                'use_asymmetric_thresholds': False,
                'long_threshold': 0.52,
                'short_threshold': 0.52
            }
        elif isinstance(config, str):
            # Load from file
            with open(config, 'r') as f:
                full_config = yaml.safe_load(f)
                self.config = full_config.get('decision', {})
        else:
            self.config = config
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters."""
        valid_sources = ['final', 'integrated']
        if self.config['signal_source'] not in valid_sources:
            raise ValueError(f"signal_source must be one of {valid_sources}, got {self.config['signal_source']}")
        
        valid_modes = ['median', 'mean', 'fixed']
        if self.config['meta_threshold_mode'] not in valid_modes:
            raise ValueError(f"meta_threshold_mode must be one of {valid_modes}, got {self.config['meta_threshold_mode']}")
        
        if self.config['meta_threshold_mode'] == 'fixed' and self.config['meta_threshold_value'] is None:
            raise ValueError("meta_threshold_value must be provided when meta_threshold_mode is 'fixed'")
    
    def generate_signal(
        self,
        direction_soft_signal: Union[np.ndarray, pd.Series],
        meta_prob: Union[np.ndarray, pd.Series] = None,
        meta_mgmt_prob: Union[np.ndarray, pd.Series] = None,
        df: pd.DataFrame = None
    ) -> np.ndarray:
        """
        Generate trading signal based on configuration.
        
        Args:
            direction_soft_signal: Soft signal from DirectionModel
            meta_prob: Meta probability from MetaFilter (required for signal_source='final')
            meta_mgmt_prob: Meta management probability from Dynamic Ensemble (required for signal_source='integrated')
            df: DataFrame with all required columns (alternative to passing arrays)
            
        Returns:
            Signal array with values: 1 (Long), -1 (Short), 0 (Flat)
        """
        # Extract from DataFrame if provided
        if df is not None:
            direction_soft_signal = df['direction_soft_signal'].values
            if 'meta_prob' in df.columns:
                meta_prob = df['meta_prob'].values
            if 'meta_mgmt_prob' in df.columns:
                meta_mgmt_prob = df['meta_mgmt_prob'].values
        
        # Apply asymmetric thresholds if configured (roadmap feature)
        if self.config['use_asymmetric_thresholds']:
            if meta_prob is None:
                raise ValueError("meta_prob is required when use_asymmetric_thresholds is True")
            
            return apply_asymmetric_thresholds(
                direction_soft_signal=direction_soft_signal,
                meta_prob=meta_prob,
                long_threshold=self.config['long_threshold'],
                short_threshold=self.config['short_threshold'],
                meta_threshold_mode=self.config['meta_threshold_mode'],
                meta_threshold_value=self.config['meta_threshold_value']
            )
        
        # Select signal source
        if self.config['signal_source'] == 'final':
            if meta_prob is None:
                raise ValueError("meta_prob is required when signal_source='final'")
            
            return compute_final_signal(
                direction_soft_signal=direction_soft_signal,
                meta_prob=meta_prob,
                meta_threshold_mode=self.config['meta_threshold_mode'],
                meta_threshold_value=self.config['meta_threshold_value'],
                direction_threshold=self.config['direction_threshold']
            )
        
        elif self.config['signal_source'] == 'integrated':
            if meta_mgmt_prob is None:
                raise ValueError("meta_mgmt_prob is required when signal_source='integrated'")
            
            return compute_integrated_signal(
                direction_soft_signal=direction_soft_signal,
                meta_mgmt_prob=meta_mgmt_prob,
                meta_threshold_mode=self.config['meta_threshold_mode'],
                meta_threshold_value=self.config['meta_threshold_value'],
                direction_threshold=self.config['direction_threshold']
            )
    
    def add_signal_to_df(
        self,
        df: pd.DataFrame,
        signal_column: str = 'signal'
    ) -> pd.DataFrame:
        """
        Add generated signal to DataFrame.
        
        Args:
            df: DataFrame with required columns
            signal_column: Name of the signal column to add
            
        Returns:
            DataFrame with signal column added
        """
        df = df.copy()
        
        # Generate signal
        signal = self.generate_signal(df=df)
        
        # Add to DataFrame
        df[signal_column] = signal
        
        return df
    
    def get_signal_stats(self, signal: np.ndarray) -> Dict[str, Any]:
        """
        Get statistics about generated signal.
        
        Args:
            signal: Signal array
            
        Returns:
            Dictionary with signal statistics
        """
        unique, counts = np.unique(signal, return_counts=True)
        stats = dict(zip(unique, counts))
        
        total = len(signal)
        
        return {
            'total_signals': total,
            'long_signals': stats.get(1, 0),
            'short_signals': stats.get(-1, 0),
            'flat_signals': stats.get(0, 0),
            'long_pct': stats.get(1, 0) / total * 100,
            'short_pct': stats.get(-1, 0) / total * 100,
            'flat_pct': stats.get(0, 0) / total * 100
        }
    
    def update_config(self, config: Dict[str, Any]):
        """
        Update configuration parameters.
        
        Args:
            config: Dictionary with configuration parameters to update
        """
        self.config.update(config)
        self._validate_config()
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get current configuration.
        
        Returns:
            Configuration dictionary
        """
        return self.config.copy()
