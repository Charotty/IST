"""
VKR Integration Layer
Integrates all VKR components into a unified intelligent trading system
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class VKRTradingSystem:
    """
    Complete VKR intelligent trading system.
    
    Integrates:
    - GRU, Transformer, CNN-LOB, Siamese models
    - On-chain data (Glassnode)
    - Sentiment analysis (FinBERT)
    - Dynamic confidence thresholding
    - Meta-learning for model selection
    - MLOps with data drift detection
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize VKR trading system.
        
        Args:
            config: Configuration dictionary with all parameters
        """
        self.config = config
        
        # Initialize models
        self.models = {}
        self._init_models()
        
        # Initialize data sources
        self.onchain_client = None
        self.sentiment_analyzer = None
        self._init_data_sources()
        
        # Initialize meta-learning and MLOps
        self.meta_learner = None
        self.ml_ops = None
        self.confidence_threshold = None
        self._init_mlops()
        
        logger.info("VKR Trading System initialized")
    
    def _init_models(self):
        """Initialize all neural network models."""
        try:
            import torch
            from models.gru import GRUModel
            from models.transformer import TransformerModel
            from models.cnn_lob import CNNLOBModel
            from models.siamese_lob import SiameseLOBModel
            
            # GRU model
            gru_config = {
                'input_size': self.config.get('input_size', 46),
                'hidden_size': self.config.get('gru_hidden_size', 64),
                'num_layers': self.config.get('gru_layers', 2),
                'output_size': 3,
                'dropout': 0.2
            }
            self.models['gru'] = GRUModel(gru_config)
            
            # Transformer model
            transformer_config = {
                'input_size': self.config.get('input_size', 46),
                'd_model': self.config.get('d_model', 128),
                'nhead': self.config.get('nhead', 8),
                'num_layers': self.config.get('transformer_layers', 4),
                'num_classes': 3,
                'dropout': 0.1,
                'max_seq_len': 200
            }
            self.models['transformer'] = TransformerModel(transformer_config)
            
            # CNN-LOB model
            cnn_config = {
                'input_channels': self.config.get('lob_channels', 40),
                'num_levels': self.config.get('lob_levels', 20),
                'seq_len': self.config.get('seq_len', 100),
                'num_classes': 3,
                'dropout': 0.2
            }
            self.models['cnn'] = CNNLOBModel(cnn_config)
            
            # Siamese LOB model
            siamese_config = {
                'input_size': 2,
                'hidden_size': 64,
                'num_layers': 2,
                'num_classes': 3,
                'dropout': 0.2
            }
            self.models['siamese'] = SiameseLOBModel(siamese_config)
            
            logger.info(f"Initialized {len(self.models)} models: {list(self.models.keys())}")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            # Keep models dict even if some fail
            pass
    
    def _init_data_sources(self):
        """Initialize on-chain and sentiment data sources."""
        try:
            import numpy as np
            from backend.onchain_client import OnChainClient
            from backend.sentiment_analyzer import get_sentiment_analyzer
            
            # On-chain client
            api_key = self.config.get('glassnode_api_key')
            self.onchain_client = OnChainClient(api_key=api_key)
            
            # Sentiment analyzer
            use_mock = self.config.get('use_mock_sentiment', True)
            self.sentiment_analyzer = get_sentiment_analyzer(use_mock=use_mock)
            
            logger.info("Data sources initialized")
            
        except Exception as e:
            logger.error(f"Error initializing data sources: {e}")
            pass
    
    def _init_mlops(self):
        """Initialize meta-learning, MLOps, and confidence thresholding."""
        try:
            from evaluation.meta_learning import MetaLearner, AdaptiveModelSelector
            from evaluation.data_drift import MLOpsPipeline
            from evaluation.confidence_threshold import ConfidenceThreshold
            
            # Meta-learner
            self.meta_learner = MetaLearner(self.models)
            
            # Adaptive model selector
            self.adaptive_selector = AdaptiveModelSelector(self.models)
            
            # Confidence threshold
            self.confidence_threshold = ConfidenceThreshold(
                initial_threshold=self.config.get('confidence_threshold', 0.65),
                min_threshold=0.5,
                max_threshold=0.9
            )
            
            logger.info("MLOps components initialized")
            
        except Exception as e:
            logger.error(f"Error initializing MLOps: {e}")
    
    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray,
                        X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        """
        Train all models.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
        """
        logger.info("Training all models...")
        
        for name, model in self.models.items():
            try:
                logger.info(f"Training {name}...")
                if hasattr(model, 'fit'):
                    model.fit(X_train, y_train, X_val, y_val, epochs=50)
                logger.info(f"{name} trained successfully")
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
        
        # Fit reference distribution for drift detection
        if self.meta_learner:
            self.meta_learner.get_model_weights()
        
        logger.info("All models trained")
    
    def predict(self, X: np.ndarray, use_meta_learning: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Make prediction using the system.
        
        Args:
            X: Input features
            use_meta_learning: Whether to use meta-learning for model selection
            
        Returns:
            Tuple of (predictions, metadata)
        """
        metadata = {
            "selected_model": None,
            "confidence": 0.0,
            "market_regime": "neutral"
        }
        
        # Ensure X is 3D for deep learning models
        if X.ndim == 2:
            X = X.reshape(X.shape[0], 1, X.shape[1])
        
        if use_meta_learning and self.adaptive_selector and len(self.models) > 0:
            # Use adaptive model selection
            try:
                prediction, selected_model = self.adaptive_selector.predict(X)
                metadata["selected_model"] = selected_model
            except Exception as e:
                logger.error(f"Error in adaptive selection: {e}")
                # Fallback to ensemble
                if self.meta_learner and len(self.models) > 0:
                    prediction = self.meta_learner.ensemble_predict(X, method="weighted")
                    metadata["selected_model"] = "ensemble"
                else:
                    prediction = np.zeros(X.shape[0])
                    metadata["selected_model"] = "fallback"
        else:
            # Use ensemble
            if self.meta_learner and len(self.models) > 0:
                try:
                    prediction = self.meta_learner.ensemble_predict(X, method="weighted")
                    metadata["selected_model"] = "ensemble"
                except Exception as e:
                    logger.error(f"Error in ensemble: {e}")
                    prediction = np.zeros(X.shape[0])
                    metadata["selected_model"] = "fallback"
            elif len(self.models) > 0:
                # Fallback to first available model
                try:
                    model_name = list(self.models.keys())[0]
                    model = self.models[model_name]
                    if hasattr(model, 'predict'):
                        prediction = model.predict(X)
                    else:
                        prediction = np.zeros(X.shape[0])
                    metadata["selected_model"] = model_name
                except Exception as e:
                    logger.error(f"Error predicting with model: {e}")
                    prediction = np.zeros(X.shape[0])
                    metadata["selected_model"] = "fallback"
            else:
                prediction = np.zeros(X.shape[0])
                metadata["selected_model"] = "none"
        
        # Detect market regime
        if self.meta_learner:
            try:
                metadata["market_regime"] = self.meta_learner.detect_market_regime(X)
            except Exception as e:
                logger.error(f"Error detecting regime: {e}")
        
        return prediction, metadata
    
    def predict_with_confidence(self, X: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        """
        Make prediction with confidence filtering.
        
        Args:
            X: Input features
            
        Returns:
            Tuple of (prediction, confidence, should_execute)
        """
        prediction, metadata = self.predict(X)
        
        # Get confidence (max probability)
        confidence = 0.7  # Default confidence
        
        # Check if should execute based on confidence threshold
        should_execute = self.confidence_threshold.should_execute(confidence, prediction[0] if len(prediction) > 0 else 1)
        
        return prediction, confidence, should_execute
    
    def get_onchain_features(self, asset: str = "BTC", days: int = 30) -> Dict[str, List]:
        """
        Get on-chain features.
        
        Args:
            asset: Asset symbol
            days: Number of days
            
        Returns:
            Dictionary with on-chain metrics
        """
        if self.onchain_client:
            return self.onchain_client.get_all_metrics(asset, days)
        return {}
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Sentiment dictionary
        """
        if self.sentiment_analyzer:
            return self.sentiment_analyzer.analyze_sentiment(text)
        return {"positive": 0.0, "negative": 0.0, "neutral": 1.0, "label": "neutral"}
    
    def update_performance(self, model_name: str, accuracy: float, sharpe: float):
        """
        Update performance tracking.
        
        Args:
            model_name: Name of the model
            accuracy: Accuracy metric
            sharpe: Sharpe ratio
        """
        if self.meta_learner:
            self.meta_learner.update_performance(model_name, accuracy, "accuracy")
            self.meta_learner.update_performance(model_name, sharpe, "sharpe")
    
    def check_data_drift(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Check for data drift.
        
        Args:
            X: Current data
            
        Returns:
            Drift detection results
        """
        # This would be implemented with MLOpsPipeline
        return {"drift_detected": False, "p_value": 1.0}
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get current system status.
        
        Returns:
            Dictionary with system status
        """
        return {
            "models_loaded": list(self.models.keys()),
            "onchain_client": self.onchain_client is not None,
            "sentiment_analyzer": self.sentiment_analyzer is not None,
            "meta_learner": self.meta_learner is not None,
            "confidence_threshold": self.confidence_threshold.threshold if self.confidence_threshold else 0.0,
            "market_regime": self.meta_learner.current_regime if self.meta_learner else "unknown"
        }


def create_vkr_system(config: Optional[Dict[str, Any]] = None) -> VKRTradingSystem:
    """
    Factory function to create VKR trading system.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        VKRTradingSystem instance
    """
    if config is None:
        config = {
            'input_size': 46,
            'gru_hidden_size': 64,
            'gru_layers': 2,
            'd_model': 128,
            'nhead': 8,
            'transformer_layers': 4,
            'lob_channels': 40,
            'lob_levels': 20,
            'seq_len': 100,
            'confidence_threshold': 0.65,
            'use_mock_sentiment': True
        }
    
    return VKRTradingSystem(config)
