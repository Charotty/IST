"""
Real-time Inference Module

Real-time inference orchestration pipeline.
Pipeline: Features → Regime Detection → Model Routing → Specialized Prediction → Meta Filter → Sizing → Execution
"""

import pandas as pd
from typing import Dict, Any, Optional
import numpy as np


class InferenceEngine:
    """
    Real-time inference engine for adaptive model selection.
    """
    
    def __init__(self):
        """Initialize inference engine."""
        self.regime_detector = None
        self.model_router = None
        self.meta_filter = None
        self.position_sizer = None
        self.is_initialized = False
    
    def initialize(self, regime_detector, model_router, meta_filter=None, position_sizer=None):
        """
        Initialize the inference engine with components.
        
        :param regime_detector: RegimeDetector instance
        :param model_router: ModelRouter instance
        :param meta_filter: Optional meta filter
        :param position_sizer: Optional position sizer
        """
        self.regime_detector = regime_detector
        self.model_router = model_router
        self.meta_filter = meta_filter
        self.position_sizer = position_sizer
        self.is_initialized = True
    
    def predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run full inference pipeline.
        
        :param df: DataFrame with features
        :return: Dict with prediction results
        """
        if not self.is_initialized:
            raise ValueError("Inference engine not initialized. Call initialize() first.")
        
        # Step 1: Regime Detection
        regime_info = self.regime_detector.get_regime_info(df)
        
        # Step 2: Model Routing
        active_model = self.model_router.route(regime_info)
        
        if active_model is None:
            # No trade if no model is active
            return {
                'signal': 0,
                'regime': regime_info,
                'confidence': 0.0,
                'position_size': 0.0
            }
        
        # Step 3: Specialized Prediction
        prediction = active_model.predict(df)
        
        # Convert prediction to signal
        if isinstance(prediction, (pd.Series, np.ndarray)):
            pred_value = prediction[-1] if len(prediction) > 0 else 0
        else:
            pred_value = prediction
        
        # Convert probability to signal
        if pred_value > 0.6:
            signal = 1
        elif pred_value < 0.4:
            signal = -1
        else:
            signal = 0
        
        confidence = abs(pred_value - 0.5) * 2  # Normalize to 0-1
        
        # Step 4: Meta Filter (if available)
        take_trade = True
        if self.meta_filter is not None:
            meta_info = self.meta_filter.get_prediction_info(df)
            take_trade = meta_info.get('take_trade', True)
        
        if not take_trade:
            signal = 0
            confidence = 0.0
        
        # Step 5: Position Sizing (if available)
        position_size = 0.0
        if self.position_sizer is not None and signal != 0:
            volatility = df['volatility'].iloc[-1] if 'volatility' in df.columns else 1.0
            position_info = self.position_sizer.get_position_info(
                signal, confidence, volatility, regime_info.get('market_regime', 'neutral')
            )
            position_size = position_info['position_size']
        
        return {
            'signal': signal,
            'regime': regime_info,
            'confidence': confidence,
            'position_size': position_size,
            'active_model': self.model_router.get_current_regime()
        }
    
    def batch_predict(self, df: pd.DataFrame, window_size: int = 24) -> pd.DataFrame:
        """
        Run batch inference on a DataFrame.
        
        :param df: DataFrame with features
        :param window_size: Window size for rolling predictions
        :return: DataFrame with prediction columns
        """
        results = []
        
        for i in range(window_size, len(df)):
            window_df = df.iloc[i-window_size:i]
            prediction = self.predict(window_df)
            
            result = {
                'timestamp': df.index[i],
                'signal': prediction['signal'],
                'confidence': prediction['confidence'],
                'position_size': prediction['position_size'],
                'regime': prediction['regime'].get('market_regime', 'unknown'),
                'active_model': prediction['active_model']
            }
            results.append(result)
        
        return pd.DataFrame(results).set_index('timestamp')
