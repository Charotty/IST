from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np

from its_project.decision.decision import Action, Signal

logger = logging.getLogger(__name__)


class SignalGenerator:
    """Generate trading signals using regression + classification filter approach."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        
        # Regression model for price prediction
        self.regression_model = None
        self.classification_model = None
        
        # Threshold strategy
        self.threshold_strategy = config.get("threshold_strategy", "fixed")
        self.fixed_threshold = config.get("fixed_threshold", 0.7)
        self.adaptive_window = config.get("adaptive_window", 100)
        
        # Signal filtering
        self.min_signal_strength = config.get("min_signal_strength", 0.1)
        self.signal_smoothing = config.get("signal_smoothing", False)
        self.smoothing_window = config.get("smoothing_window", 5)
        
        # Historical data for adaptive thresholds
        self.prediction_history = []
        self.return_history = []
    
    def set_models(self, regression_model: Any, classification_model: Any) -> None:
        """Set the regression and classification models."""
        self.regression_model = regression_model
        self.classification_model = classification_model
    
    def generate_signal(
        self,
        features: np.ndarray,
        current_price: float,
        timestamp_ms: int,
        symbol: str = "BTCUSDT"
    ) -> Optional[Signal]:
        """
        Generate trading signal using regression + classification filter.
        
        Args:
            features: Feature array
            current_price: Current market price
            timestamp_ms: Timestamp in milliseconds
            symbol: Trading symbol
            
        Returns:
            Trading signal or None if no signal
        """
        if self.regression_model is None or self.classification_model is None:
            logger.warning("Models not set. Call set_models() first.")
            return None
        
        try:
            # Step 1: Get regression prediction (price change)
            price_change = self._predict_price_change(features)
            
            # Step 2: Get classification prediction (direction confidence)
            class_proba = self.classification_model.predict_proba(features)
            class_pred = self.classification_model.predict(features)
            
            # Step 3: Apply classification filter
            if not self._classification_filter(class_pred, class_proba):
                return None
            
            # Step 4: Convert regression prediction to action
            action = self._price_change_to_action(price_change)
            
            # Step 5: Apply threshold
            threshold = self._get_threshold()
            confidence = self._calculate_confidence(price_change, class_proba)
            
            if confidence < threshold:
                return None
            
            # Step 6: Create signal
            signal = Signal(
                action=action,
                confidence=confidence,
                timestamp_ms=timestamp_ms,
                symbol=symbol,
                metadata={
                    "price_change": price_change,
                    "classification": int(class_pred[0]),
                    "threshold": threshold,
                    "threshold_strategy": self.threshold_strategy
                }
            )
            
            # Update history
            self._update_history(price_change, confidence)
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signal: {e}")
            return None
    
    def _predict_price_change(self, features: np.ndarray) -> float:
        """Predict price change using regression model."""
        if hasattr(self.regression_model, 'predict'):
            prediction = self.regression_model.predict(features)
            return float(prediction[0]) if hasattr(prediction, '__iter__') else float(prediction)
        else:
            raise ValueError("Regression model must have predict method")
    
    def _classification_filter(self, class_pred: np.ndarray, class_proba: np.ndarray) -> bool:
        """Filter signals based on classification confidence."""
        max_confidence = np.max(class_proba)
        
        # Filter out low confidence predictions
        if max_confidence < self.min_signal_strength:
            return False
        
        # Filter out hold signals (class 1) unless very strong
        if class_pred[0] == 1 and max_confidence < 0.9:
            return False
        
        return True
    
    def _price_change_to_action(self, price_change: float) -> Action:
        """Convert price change prediction to trading action."""
        # Define thresholds for price changes
        strong_change_threshold = 0.002  # 0.2%
        weak_change_threshold = 0.0005  # 0.05%
        
        if price_change > strong_change_threshold:
            return Action.BUY
        elif price_change < -strong_change_threshold:
            return Action.SELL
        elif abs(price_change) > weak_change_threshold:
            # Weak signals - could be HOLD or based on other factors
            return Action.HOLD
        else:
            return Action.HOLD
    
    def _get_threshold(self) -> float:
        """Get confidence threshold based on strategy."""
        if self.threshold_strategy == "fixed":
            return self.fixed_threshold
        elif self.threshold_strategy == "adaptive":
            return self._adaptive_threshold()
        elif self.threshold_strategy == "volatility_adjusted":
            return self._volatility_adjusted_threshold()
        else:
            logger.warning(f"Unknown threshold strategy: {self.threshold_strategy}")
            return self.fixed_threshold
    
    def _adaptive_threshold(self) -> float:
        """Adaptive threshold based on recent performance."""
        if len(self.prediction_history) < self.adaptive_window:
            return self.fixed_threshold
        
        recent_predictions = self.prediction_history[-self.adaptive_window:]
        recent_returns = self.return_history[-self.adaptive_window:]
        
        # Calculate recent success rate
        correct_predictions = sum(1 for pred, ret in zip(recent_predictions, recent_returns) 
                                if (pred > 0 and ret > 0) or (pred < 0 and ret < 0))
        success_rate = correct_predictions / len(recent_predictions)
        
        # Adjust threshold based on success rate
        if success_rate > 0.6:
            # Lower threshold for more signals when performing well
            return max(0.5, self.fixed_threshold - 0.1)
        elif success_rate < 0.4:
            # Raise threshold for fewer signals when performing poorly
            return min(0.9, self.fixed_threshold + 0.1)
        else:
            return self.fixed_threshold
    
    def _volatility_adjusted_threshold(self) -> float:
        """Adjust threshold based on market volatility."""
        if len(self.return_history) < 20:
            return self.fixed_threshold
        
        recent_returns = self.return_history[-20:]
        volatility = np.std(recent_returns)
        
        # Higher volatility -> lower threshold (more opportunities)
        # Lower volatility -> higher threshold (fewer false signals)
        volatility_factor = min(2.0, max(0.5, volatility / 0.01))  # Normalize around 1% volatility
        
        adjusted_threshold = self.fixed_threshold / volatility_factor
        return max(0.3, min(0.9, adjusted_threshold))
    
    def _calculate_confidence(self, price_change: float, class_proba: np.ndarray) -> float:
        """Calculate overall signal confidence."""
        # Combine regression and classification confidence
        classification_confidence = np.max(class_proba)
        
        # Convert price change magnitude to confidence
        price_magnitude = abs(price_change)
        regression_confidence = min(1.0, price_magnitude / 0.005)  # Normalize around 0.5%
        
        # Weighted combination
        total_confidence = 0.6 * classification_confidence + 0.4 * regression_confidence
        
        return float(total_confidence)
    
    def _update_history(self, price_change: float, confidence: float) -> None:
        """Update prediction history for adaptive strategies."""
        self.prediction_history.append(price_change)
        
        # Return history would be updated with actual returns
        # This should be called externally with actual market returns
        if len(self.prediction_history) > 1000:
            # Keep only recent history
            self.prediction_history = self.prediction_history[-1000:]
    
    def update_return_history(self, actual_return: float) -> None:
        """Update return history with actual market returns."""
        self.return_history.append(actual_return)
        
        if len(self.return_history) > 1000:
            self.return_history = self.return_history[-1000:]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get recent performance statistics."""
        if len(self.prediction_history) < 10:
            return {"message": "Insufficient data"}
        
        recent_predictions = self.prediction_history[-100:]
        recent_returns = self.return_history[-100:] if self.return_history else []
        
        stats = {
            "total_predictions": len(recent_predictions),
            "avg_prediction": np.mean(recent_predictions),
            "prediction_std": np.std(recent_predictions),
        }
        
        if recent_returns:
            correct = sum(1 for pred, ret in zip(recent_predictions, recent_returns) 
                         if (pred > 0 and ret > 0) or (pred < 0 and ret < 0))
            stats.update({
                "accuracy": correct / len(recent_returns),
                "total_returns": np.sum(recent_returns),
                "avg_return": np.mean(recent_returns),
            })
        
        return stats
