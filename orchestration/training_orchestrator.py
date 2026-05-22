"""
Training Orchestrator

Unified training orchestration with explicit pipeline contract.
Pipeline: features → regime → all model predictions → meta weighting → decision → risk

Supports walk-forward optimization (WFO) and train→test validation.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

from .orchestrator_config import OrchestratorConfig
from utils.data_leakage_prevention import (
    DataLeakagePreventer,
    LeakageConfig,
    compute_train_threshold,
)
from decision.decision_pipeline import DecisionPipeline


@dataclass
class TrainingResult:
    """Result from training orchestration."""
    model_predictions: Dict[str, np.ndarray]
    regime_predictions: np.ndarray
    meta_weights: Dict[str, np.ndarray]
    meta_probabilities: np.ndarray
    direction_signals: np.ndarray
    final_signals: np.ndarray
    metrics: Dict[str, Any]
    position_sizes: np.ndarray


class TrainingOrchestrator:
    """
    Unified training orchestrator with explicit pipeline contract.
    
    Pipeline:
    features[t] → regime[t] → {p_lgb, p_gru, p_xgb, p_cnn, ...}[t] → meta_mgmt[t] → direction_soft[t] → decision → risk
    """
    
    def __init__(self, config: OrchestratorConfig):
        """
        Initialize training orchestrator.
        
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
        self._runtime_train_meta_threshold: Optional[float] = None
        
        # Initialize data leakage preventer
        leakage_config = LeakageConfig(
            horizon=config.prediction_horizon,
            embargo=config.embargo_period,
            purge=config.enable_purge,
            allow_meta_label_on_test=config.allow_meta_label_on_test
        )
        self.leakage_preventer = DataLeakagePreventer(leakage_config)
        
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
            self.decision_pipeline = self._build_decision_pipeline(for_inference=False)
        else:
            self.decision_pipeline = None
        self.is_initialized = True
    
    def _build_decision_pipeline(self, for_inference: bool = False) -> DecisionPipeline:
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
    
    def collect_predictions(self, features: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Collect predictions from ALL models.
        
        This is the key difference from the old InferenceEngine:
        - Old: calls ONE model via router
        - New: calls ALL models and collects predictions
        
        Args:
            features: DataFrame with features
            
        Returns:
            Dictionary of predictions {model_key: prediction_array}
        """
        predictions = {}
        
        for model_key, model in self.models.items():
            pred = model.predict(features)
            
            # Convert to numpy array if needed
            if isinstance(pred, (pd.Series, pd.DataFrame)):
                pred = pred.values
            
            predictions[model_key] = pred
        
        return predictions
    
    def run_pipeline(
        self,
        features: pd.DataFrame,
        return_intermediate: bool = False
    ) -> TrainingResult:
        """
        Run full training pipeline with explicit contract.
        
        Pipeline:
        features[t] → regime[t] → {p_lgb, p_gru, p_xgb, p_cnn, ...}[t] → meta_mgmt[t] → direction_soft[t] → decision → risk
        
        Args:
            features: DataFrame with features
            return_intermediate: Whether to return intermediate results
            
        Returns:
            TrainingResult with all pipeline outputs
        """
        if not self.is_initialized:
            raise ValueError("TrainingOrchestrator not initialized. Call initialize() first.")
        
        # Step 1: Regime Detection
        regime_info = self.regime_detector.get_regime_info(features)
        regime_pred = regime_info.get('regime_pred', np.zeros(len(features)))
        regime_pred = np.asarray(regime_pred).reshape(-1)
        if regime_pred.shape[0] != len(features):
            regime_pred = np.resize(regime_pred, len(features))
        
        # Step 2: Collect predictions from ALL models
        model_predictions = self.collect_predictions(features)
        
        # Step 3: Apply meta weighting (regime-aware ensemble)
        meta_weights = self.meta_weighting.get_weights(regime_pred, mode=self.config.ensemble_mode)
        meta_probabilities = self.meta_weighting.apply_dynamic_weighting(
            model_predictions, regime_pred, mode=self.config.ensemble_mode
        )
        
        # Step 4: Convert to direction signals (probability → discrete legs)
        direction_signals = np.where(
            meta_probabilities > self.config.direction_threshold,
            1,
            np.where(meta_probabilities < (1 - self.config.direction_threshold), -1, 0)
        )
        
        soft = meta_probabilities.astype(float)
        
        # Step 5: Decision layer — DecisionPipeline (integrated) or legacy decision_engine
        if self.decision_pipeline is not None:
            final_signals = self.decision_pipeline.generate_signal(
                direction_soft_signal=soft,
                meta_mgmt_prob=soft,
                train_threshold_override=self._runtime_train_meta_threshold,
            )
        elif self.decision_engine is not None:
            final_signals = self.decision_engine.make_decision(
                direction_signals, meta_probabilities, regime_info
            )
        else:
            final_signals = direction_signals

        if getattr(self.config, "signal_strategy", "ensemble") == "momentum_confirm":
            final_signals = self._momentum_confirm_signals(features, meta_probabilities)

        final_signals = self._apply_signal_margin(final_signals, meta_probabilities)
        final_signals = self._apply_trade_mode(final_signals)
        final_signals = self._apply_volatility_filter(final_signals, features)

        # Step 6: Apply risk management (if available)
        if self.risk_manager is not None:
            position_sizes = self.risk_manager.calculate_position_sizes(
                final_signals, meta_probabilities, features
            )
            position_sizes = np.asarray(position_sizes, dtype=float).reshape(-1)
            if position_sizes.shape[0] != len(final_signals):
                raise ValueError("risk_manager.calculate_position_sizes must return one value per row")
        else:
            position_sizes = np.ones(len(final_signals), dtype=float)
        
        # Calculate metrics
        metrics = self._calculate_metrics(
            model_predictions, meta_probabilities, direction_signals, final_signals
        )
        
        return TrainingResult(
            model_predictions=model_predictions,
            regime_predictions=regime_pred,
            meta_weights=meta_weights,
            meta_probabilities=meta_probabilities,
            direction_signals=direction_signals,
            final_signals=final_signals,
            metrics=metrics,
            position_sizes=position_sizes,
        )
    
    def walk_forward_optimization(
        self,
        features: pd.DataFrame,
        targets: pd.Series
    ) -> List[TrainingResult]:
        """
        Run walk-forward optimization (WFO) with data leakage prevention.
        
        Args:
            features: DataFrame with features
            targets: Series with targets
            
        Returns:
            List of TrainingResult for each fold
        """
        if not self.is_initialized:
            raise ValueError("TrainingOrchestrator not initialized. Call initialize() first.")
        
        results = []
        
        # Use safe walk-forward split with purge and embargo
        splits = self.leakage_preventer.safe_walk_forward_split(
            features, targets,
            self.config.train_window_size,
            self.config.test_window_size,
            self.config.walk_forward_step,
            self.config.prediction_horizon
        )
        
        for train_features, train_targets, test_features, test_targets in splits:
            # Validate no leakage
            validation = self.leakage_preventer.validate_no_leakage(
                train_features, test_features, train_targets, test_targets
            )
            
            if not validation['validation_passed']:
                raise ValueError(
                    f"Data leakage detected in WFO fold: {validation}. "
                    "This will cause inflated metrics in training but degradation in live trading."
                )
            
            # Train all models (drop NaN targets from training)
            train_mask = ~train_targets.isna()
            train_features_clean = train_features[train_mask]
            train_targets_clean = train_targets[train_mask]
            
            for model_key, model in self.models.items():
                if hasattr(model, 'fit'):
                    model.fit(train_features_clean, train_targets_clean)
            
            # Run pipeline: calibrate meta threshold on train meta_probs only, apply to test
            train_result = self.run_pipeline(train_features)
            self._runtime_train_meta_threshold = self._calibrate_test_meta_threshold(
                train_result.meta_probabilities
            )
            test_result = self.run_pipeline(test_features)
            self._runtime_train_meta_threshold = None
            results.append(test_result)
        
        return results
    
    def train_test_split(
        self,
        features: pd.DataFrame,
        targets: pd.Series,
        test_ratio: float = 0.2
    ) -> Tuple[TrainingResult, TrainingResult]:
        """
        Run simple train→test validation with data leakage prevention.
        
        Args:
            features: DataFrame with features
            targets: Series with targets
            test_ratio: Ratio of test data
            
        Returns:
            Tuple of (train_result, test_result)
        """
        if not self.is_initialized:
            raise ValueError("TrainingOrchestrator not initialized. Call initialize() first.")
        
        # Use safe train/test split with purge and embargo
        train_features, train_targets, test_features, test_targets = \
            self.leakage_preventer.safe_train_test_split(
                features, targets, test_ratio, self.config.prediction_horizon
            )
        
        # Validate no leakage
        validation = self.leakage_preventer.validate_no_leakage(
            train_features, test_features, train_targets, test_targets
        )
        
        if not validation['validation_passed']:
            raise ValueError(
                f"Data leakage detected: {validation}. "
                "This will cause inflated metrics in training but degradation in live trading."
            )
        
        # Train all models (drop NaN targets from training)
        train_mask = ~train_targets.isna()
        train_features_clean = train_features[train_mask]
        train_targets_clean = train_targets[train_mask]
        
        for model_key, model in self.models.items():
            if hasattr(model, 'fit'):
                model.fit(train_features_clean, train_targets_clean)
        
        # Run pipeline: calibrate meta threshold on train only for test evaluation
        train_result = self.run_pipeline(train_features)
        self._runtime_train_meta_threshold = self._calibrate_test_meta_threshold(
            train_result.meta_probabilities
        )
        test_result = self.run_pipeline(test_features)
        self._runtime_train_meta_threshold = None
        
        return train_result, test_result
    
    def _momentum_confirm_signals(
        self, features: pd.DataFrame, meta_probabilities: np.ndarray
    ) -> np.ndarray:
        """Long when price > SMA and meta prob confirms; flat otherwise."""
        period = int(getattr(self.config, "momentum_sma_period", 100))
        close = features["close"].astype(float)
        sma = close.rolling(period, min_periods=1).mean()
        trend_up = (close > sma).to_numpy()
        p = np.asarray(meta_probabilities, dtype=float).reshape(-1)
        thr = float(self.config.direction_threshold)
        sig = np.zeros(len(p), dtype=float)
        sig[trend_up & (p > thr)] = 1.0
        if self.config.trade_mode == "both":
            sig[(~trend_up) & (p < (1.0 - thr))] = -1.0
        return sig

    def _apply_signal_margin(
        self, signals: np.ndarray, meta_probabilities: np.ndarray
    ) -> np.ndarray:
        margin = float(getattr(self.config, "min_signal_margin", 0.0) or 0.0)
        if margin <= 0:
            return np.asarray(signals, dtype=float).reshape(-1)
        s = np.asarray(signals, dtype=float).reshape(-1)
        p = np.asarray(meta_probabilities, dtype=float).reshape(-1)
        weak = np.abs(p - 0.5) < margin
        s = s.copy()
        s[weak] = 0.0
        return s

    def _apply_trade_mode(self, signals: np.ndarray) -> np.ndarray:
        mode = getattr(self.config, "trade_mode", "both")
        s = np.asarray(signals, dtype=float).reshape(-1)
        if mode == "long_only":
            return np.where(s > 0, s, 0.0)
        if mode == "short_only":
            return np.where(s < 0, s, 0.0)
        return s

    def _atr_series(self, features: pd.DataFrame) -> np.ndarray:
        if "atr" in features.columns:
            return features["atr"].astype(float).to_numpy()
        if "volatility" in features.columns and "close" in features.columns:
            return (features["volatility"].astype(float) * features["close"].astype(float)).to_numpy()
        close = features["close"].astype(float)
        tr = close.pct_change().abs()
        return (tr.rolling(14, min_periods=1).mean() * close).bfill().ffill().to_numpy()

    def _compute_atr_threshold(self, train_features: pd.DataFrame) -> Optional[float]:
        pct = float(getattr(self.config, "volatility_filter_percentile", 0.0) or 0.0)
        if pct <= 0:
            return None
        atr = self._atr_series(train_features)
        finite = atr[np.isfinite(atr)]
        if len(finite) == 0:
            return None
        return float(np.percentile(finite, pct))

    def _apply_volatility_filter(
        self, signals: np.ndarray, features: pd.DataFrame
    ) -> np.ndarray:
        thr = getattr(self, "_volatility_atr_threshold", None)
        if thr is None:
            return np.asarray(signals, dtype=float).reshape(-1)
        s = np.asarray(signals, dtype=float).reshape(-1).copy()
        atr = self._atr_series(features)
        s[atr > thr] = 0.0
        return s

    def _calibrate_test_meta_threshold(self, train_meta_probs: np.ndarray) -> Optional[float]:
        """Train-only scalar meta threshold for OOS application (no test leakage)."""
        if self.config.meta_threshold_mode == "fixed":
            return float(self.config.meta_threshold)
        mode = self.config.meta_threshold_mode
        if mode not in ("median", "mean", "percentile"):
            mode = "median"
        return compute_train_threshold(np.asarray(train_meta_probs), mode=mode)
    
    def walk_forward_backtest(
        self,
        features: pd.DataFrame,
        targets: pd.Series,
        *,
        commission: Optional[float] = None,
        slippage: Optional[float] = None,
        on_fold_done: Optional[Any] = None,
    ) -> pd.DataFrame:
        """
        Walk-forward (leakage-safe) training per fold, then OOS backtest on ``Backtester``
        using the same ``final_signals`` and ``position_sizes`` as ``run_pipeline``.
        """
        if not self.is_initialized:
            raise ValueError("TrainingOrchestrator not initialized. Call initialize() first.")
        from backtesting.backtester import Backtester
        from backtesting.metrics_config import load_backtesting_config
        from backtesting.performance_metrics import PerformanceMetrics

        bt_cfg = load_backtesting_config()
        m_cfg = bt_cfg.metrics
        bt = Backtester(
            commission=commission if commission is not None else m_cfg.commission,
            slippage=slippage if slippage is not None else m_cfg.slippage,
        )
        rows: List[Dict[str, Any]] = []

        splits = self.leakage_preventer.safe_walk_forward_split(
            features,
            targets,
            self.config.train_window_size,
            self.config.test_window_size,
            self.config.walk_forward_step,
            self.config.prediction_horizon,
        )
        cap = int(getattr(self.config, "max_wfo_folds", 0) or 0)
        if cap > 0:
            splits = splits[:cap]

        for fold_idx, (train_features, train_targets, test_features, test_targets) in enumerate(splits):
            validation = self.leakage_preventer.validate_no_leakage(
                train_features, test_features, train_targets, test_targets
            )
            if not validation["validation_passed"]:
                raise ValueError(f"Data leakage detected in WFO fold: {validation}")

            train_mask = ~train_targets.isna()
            train_features_clean = train_features[train_mask]
            train_targets_clean = train_targets[train_mask]

            for model_key, model in self.models.items():
                if hasattr(model, "fit"):
                    model.fit(train_features_clean, train_targets_clean)

            self._volatility_atr_threshold = self._compute_atr_threshold(train_features)
            train_result = self.run_pipeline(train_features)
            self._runtime_train_meta_threshold = self._calibrate_test_meta_threshold(
                train_result.meta_probabilities
            )
            test_result = self.run_pipeline(test_features)
            self._runtime_train_meta_threshold = None
            self._volatility_atr_threshold = None

            if "close" not in test_features.columns:
                raise ValueError("walk_forward_backtest requires a 'close' column in features for Backtester")

            train_perf = bt.run(
                train_features[["close"]],
                train_result.final_signals,
                position_size=train_result.position_sizes,
            )
            test_perf = bt.run(
                test_features[["close"]],
                test_result.final_signals,
                position_size=test_result.position_sizes,
            )
            pm_is = PerformanceMetrics(train_perf, m_cfg)
            pm_oos = PerformanceMetrics(test_perf, m_cfg)
            is_ann = pm_is.annualized_return_from_results()
            oos_ann = pm_oos.annualized_return_from_results()
            wfe = (oos_ann / is_ann) if is_ann not in (0.0, -0.0) and np.isfinite(is_ann) else float("nan")

            metrics = pm_oos.calculate_metrics().to_dict()
            for k, v in pm_is.calculate_metrics().to_dict().items():
                metrics[f"IS_{k}"] = v
            metrics["Walk-Forward Efficiency"] = wfe
            metrics["IS Annualized Return"] = is_ann
            metrics["OOS Annualized Return"] = oos_ann
            metrics["Fold"] = fold_idx + 1
            rows.append(metrics)
            if on_fold_done is not None:
                fold_rows = [dict(r) for r in rows]
                if on_fold_done(fold_rows):
                    break

        return pd.DataFrame(rows)
    
    def _calculate_metrics(
        self,
        model_predictions: Dict[str, np.ndarray],
        meta_probabilities: np.ndarray,
        direction_signals: np.ndarray,
        final_signals: np.ndarray
    ) -> Dict[str, Any]:
        """Calculate metrics for training results."""
        metrics = {
            'meta_prob_mean': float(np.mean(meta_probabilities)),
            'meta_prob_std': float(np.std(meta_probabilities)),
            'direction_signal_mean': float(np.mean(direction_signals)),
            'final_signal_mean': float(np.mean(final_signals)),
            'signal_count': int(np.sum(final_signals != 0)),
        }
        
        # Model-specific metrics
        for model_key, pred in model_predictions.items():
            metrics[f'{model_key}_prob_mean'] = float(np.mean(pred))
            metrics[f'{model_key}_prob_std'] = float(np.std(pred))
        
        return metrics
    
    def get_model_keys(self) -> List[str]:
        """Get fixed model keys from config."""
        return self.config.model_keys.copy()
    
    def get_regime_keys(self) -> List[str]:
        """Get fixed regime keys from config."""
        return self.config.regime_keys.copy()
