"""
Inference Orchestrator

Unified inference orchestration with explicit pipeline contract.
Pipeline: features → regime → all model predictions → meta weighting → decision → risk

This module is the **canonical** multi-model inference entry point for production (all models
→ regime-aware ensemble → `DecisionPipeline` integrated signal).

For the legacy single-model router path, see `models/inference/inference_engine.py`
(`InferenceEngine` + `ModelRouter`); do not mix both entry points in one deployment without
an explicit adapter.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from dataclasses import dataclass

from .orchestrator_config import OrchestratorConfig
from decision.decision_pipeline import DecisionPipeline


@dataclass
class InferenceResult:
    """Result from inference orchestration."""
    signal: int
    confidence: float
    position_size: float
    regime: str
    model_predictions: Dict[str, float]
    meta_probability: float
    meta_weights: Dict[str, float]
    active_models: list


class InferenceOrchestrator:
    """
    Unified inference orchestrator with explicit pipeline contract.
    
    Pipeline:
    features[t] → regime[t] → {p_lgb, p_gru, p_xgb, p_cnn, ...}[t] → meta_mgmt[t] → direction_soft[t] → decision → risk
    
    Key difference from old InferenceEngine:
    - Old: regime → router → ONE active model → threshold
    - New: regime → ALL models → meta weighting → threshold
    """
    
    def __init__(self, config: OrchestratorConfig):
        """
        Initialize inference orchestrator.
        
        Args:
            config: OrchestratorConfig with fixed model keys and parameters
        """
        self.config = config
        self.models = {}  # model_key -> model_instance
        self.regime_detector = None
        self.meta_weighting = None
        self.decision_engine = None
        self.risk_manager = None
        self.decision_pipeline: Optional[DecisionPipeline] = None
        
        self.is_initialized = False
    
    def initialize(
        self,
        models: Dict[str, Any],
        regime_detector: Any,
        meta_weighting: Any,
        decision_engine: Optional[Any] = None,
        risk_manager: Optional[Any] = None,
        decision_pipeline: Optional[DecisionPipeline] = None,
    ):
        """
        Initialize orchestrator with components.
        
        Args:
            models: Dictionary of models {model_key: model_instance}
            regime_detector: Regime detector instance
            meta_weighting: Meta weighting instance (DynamicMetaWeighting)
            decision_engine: Optional decision engine
            risk_manager: Optional risk manager
            decision_pipeline: Optional DecisionPipeline; if None and config.apply_decision_pipeline, a default is built
        """
        # Validate model keys match config
        model_keys = set(models.keys())
        config_keys = set(self.config.model_keys)
        
        if model_keys != config_keys:
            raise ValueError(
                f"Model keys {model_keys} do not match config.model_keys {config_keys}. "
                f"Ensure all models in config are registered."
            )
        
        self.models = models
        self.regime_detector = regime_detector
        self.meta_weighting = meta_weighting
        self.decision_engine = decision_engine
        self.risk_manager = risk_manager
        if decision_pipeline is not None:
            self.decision_pipeline = decision_pipeline
        elif self.config.apply_decision_pipeline:
            self.decision_pipeline = self._build_decision_pipeline(for_inference=True)
        else:
            self.decision_pipeline = None
        self.is_initialized = True
    
    def _build_decision_pipeline(self, for_inference: bool = True) -> DecisionPipeline:
        mode = self.config.meta_threshold_mode
        fixed_val = float(self.config.meta_threshold) if mode == "fixed" else None
        cfg = {
            'signal_source': 'integrated',
            'direction_threshold': self.config.direction_threshold,
            'meta_threshold_mode': 'median' if mode not in ('median', 'mean', 'fixed') else mode,
            'meta_threshold_value': fixed_val,
            'use_asymmetric_thresholds': False,
            'safe_mode': True,
            'train_threshold': float(self.config.meta_threshold) if for_inference else None,
            'threshold_window': self.config.threshold_rolling_window,
            'threshold_expanding': False,
        }
        if mode == 'fixed' and cfg['meta_threshold_value'] is None:
            cfg['meta_threshold_value'] = float(self.config.meta_threshold)
        return DecisionPipeline(cfg)
    
    def collect_predictions(self, features: pd.DataFrame) -> Dict[str, float]:
        """
        Collect predictions from ALL models for single timestep.
        
        Args:
            features: DataFrame with features (single timestep or window)
            
        Returns:
            Dictionary of predictions {model_key: prediction_value}
        """
        predictions = {}
        
        for model_key, model in self.models.items():
            # Try predict_single first, fall back to predict
            if hasattr(model, 'predict_single'):
                pred_value = model.predict_single(features)
            else:
                pred = model.predict(features)
                
                # Extract single value - handle different return types
                if isinstance(pred, (pd.Series, np.ndarray)):
                    # Get the last value (most recent prediction)
                    pred_value = pred.flatten()[-1] if len(pred) > 0 else 0.5
                else:
                    pred_value = pred
            
            # Ensure it's a float
            predictions[model_key] = float(pred_value)
        
        return predictions
    
    def predict(self, features: pd.DataFrame) -> InferenceResult:
        """
        Run full inference pipeline with explicit contract.
        
        Pipeline:
        features[t] → regime[t] → {p_lgb, p_gru, p_xgb, p_cnn, ...}[t] → meta_mgmt[t] → direction_soft[t] → decision → risk
        
        Args:
            features: DataFrame with features (single timestep or window)
            
        Returns:
            InferenceResult with all pipeline outputs
        """
        if not self.is_initialized:
            raise ValueError("InferenceOrchestrator not initialized. Call initialize() first.")
        
        # Step 1: Regime Detection
        regime_info = self.regime_detector.get_regime_info(features)
        regime_name = regime_info.get('market_regime', 'unknown')
        regime_pred_array = regime_info.get('regime_pred', 0)  # 1=trend, 0=range
        
        # Extract single regime prediction (last value)
        if isinstance(regime_pred_array, np.ndarray):
            regime_pred = int(regime_pred_array[-1]) if len(regime_pred_array) > 0 else 0
        else:
            regime_pred = int(regime_pred_array)
        
        # Step 2: Collect predictions from ALL models
        model_predictions = self.collect_predictions(features)
        
        # Step 3: Apply meta weighting (regime-aware ensemble)
        # Convert single values to arrays for meta weighting
        regime_pred_array = np.array([regime_pred])
        # Ensure predictions are 1D arrays of length 1
        predictions_array = {}
        for k, v in model_predictions.items():
            predictions_array[k] = np.array([float(v)])
        
        meta_weights_dict = self.meta_weighting.get_weights(
            regime_pred_array, mode=self.config.ensemble_mode
        )
        
        # Extract single weights - handle different array structures
        meta_weights = {}
        for k, v in meta_weights_dict.items():
            if isinstance(v, np.ndarray):
                if v.ndim == 1 and len(v) == 1:
                    meta_weights[k] = float(v[0])
                elif v.ndim == 0:
                    meta_weights[k] = float(v)
                else:
                    meta_weights[k] = float(v.flatten()[0])
            else:
                meta_weights[k] = float(v)
        
        meta_probability = self.meta_weighting.apply_dynamic_weighting(
            predictions_array, regime_pred_array, mode=self.config.ensemble_mode
        )[0]
        
        soft = np.array([float(meta_probability)], dtype=float)
        
        # Step 4–5: Direction / decision — DecisionPipeline (integrated) or legacy
        if self.decision_pipeline is not None:
            sig_series = self.decision_pipeline.generate_signal(
                direction_soft_signal=soft,
                meta_mgmt_prob=soft,
            )
            final_signal = int(sig_series[0]) if len(sig_series) else 0
        elif self.decision_engine is not None:
            if meta_probability > self.config.signal_threshold:
                direction_signal = 1
            elif meta_probability < (1 - self.config.signal_threshold):
                direction_signal = -1
            else:
                direction_signal = 0
            fs = self.decision_engine.make_decision(
                np.array([direction_signal]), np.array([meta_probability]), regime_info
            )
            final_signal = int(fs[0]) if hasattr(fs, '__len__') else int(fs)
        else:
            if meta_probability > self.config.signal_threshold:
                final_signal = 1
            elif meta_probability < (1 - self.config.signal_threshold):
                final_signal = -1
            else:
                final_signal = 0
        
        # Step 6: Apply risk management (if available)
        position_size = 0.0
        if self.risk_manager is not None and final_signal != 0:
            volatility = features['volatility'].iloc[-1] if 'volatility' in features.columns else 1.0
            position_info = self.risk_manager.calculate_position_size(
                final_signal, meta_probability, volatility, regime_name
            )
            position_size = position_info.get('position_size', 0.0)
        elif final_signal != 0:
            # Default position sizing
            position_size = 1.0
        
        # Calculate confidence
        confidence = abs(meta_probability - 0.5) * 2  # Normalize to 0-1
        
        return InferenceResult(
            signal=int(final_signal),
            confidence=float(confidence),
            position_size=float(position_size),
            regime=str(regime_name),
            model_predictions=model_predictions,
            meta_probability=float(meta_probability),
            meta_weights=meta_weights,
            active_models=list(self.models.keys())
        )
    
    def batch_predict(self, features: pd.DataFrame, window_size: int = 24) -> pd.DataFrame:
        """
        Run batch inference on a DataFrame.
        
        Args:
            features: DataFrame with features
            window_size: Window size for rolling predictions
            
        Returns:
            DataFrame with prediction columns
        """
        results = []
        
        for i in range(window_size, len(features)):
            window_df = features.iloc[i-window_size:i]
            prediction = self.predict(window_df)
            
            result = {
                'timestamp': features.index[i],
                'signal': prediction.signal,
                'confidence': prediction.confidence,
                'position_size': prediction.position_size,
                'regime': prediction.regime,
                'meta_probability': prediction.meta_probability,
            }
            
            # Add individual model predictions
            for model_key in self.config.model_keys:
                result[f'{model_key}_pred'] = prediction.model_predictions.get(model_key, 0.5)
            
            # Add meta weights
            for model_key in self.config.model_keys:
                result[f'{model_key}_weight'] = prediction.meta_weights.get(model_key, 0.0)
            
            results.append(result)
        
        return pd.DataFrame(results).set_index('timestamp')
    
    def get_model_keys(self) -> list:
        """Get fixed model keys from config."""
        return self.config.model_keys.copy()
    
    def get_regime_keys(self) -> list:
        """Get fixed regime keys from config."""
        return self.config.regime_keys.copy()
