"""
Model Router Module

Dynamic model orchestration layer.
Selects which predictive model should be used based on current regime conditions.
"""

from typing import Dict, Any, Optional
import pandas as pd


class ModelRouter:
    """
    Model router for dynamic model selection based on regime.
    """
    
    def __init__(self):
        """Initialize model router."""
        self.models = {}
        self.active_model = None
        self.current_regime = None
    
    def register_model(self, regime: str, model: Any):
        """
        Register a model for a specific regime.
        
        :param regime: Regime name (e.g., 'trend', 'range', 'breakout')
        :param model: Model instance
        """
        self.models[regime] = model
    
    def route(self, regime_info: Dict[str, Any]) -> Optional[Any]:
        """
        Route to appropriate model based on regime info.
        
        :param regime_info: Dict with regime information
        :return: Active model for current regime
        """
        market_regime = regime_info.get('market_regime')
        trade_allowed = regime_info.get('trade_allowed', False)
        
        if not trade_allowed:
            self.active_model = None
            self.current_regime = 'no_trade'
            return None
        
        # Map regime to model
        regime_mapping = {
            'trend': 'trend',
            'range': 'mean_reversion',
            'breakout': 'volatility'
        }
        
        model_key = regime_mapping.get(market_regime)
        
        if model_key in self.models:
            self.active_model = self.models[model_key]
            self.current_regime = market_regime
            return self.active_model
        else:
            self.active_model = None
            self.current_regime = 'unknown'
            return None
    
    def get_active_model(self) -> Optional[Any]:
        """
        Get the currently active model.
        
        :return: Active model instance
        """
        return self.active_model
    
    def get_current_regime(self) -> Optional[str]:
        """
        Get the current regime.
        
        :return: Current regime name
        """
        return self.current_regime
    
    def predict(self, df: pd.DataFrame, regime_info: Dict[str, Any]):
        """
        Route and predict using the appropriate model.
        
        :param df: DataFrame with features
        :param regime_info: Dict with regime information
        :return: Prediction from active model
        """
        model = self.route(regime_info)
        
        if model is None:
            # Return neutral prediction if no model is active
            return pd.Series(0, index=df.index)
        
        return model.predict(df)
