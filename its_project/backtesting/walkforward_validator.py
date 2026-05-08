from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta

from its_project.backtesting.base import BaseBacktester, BacktestResult
from its_project.models.base import BaseModel
from its_project.targets.economic_target import EconomicTargetCalculator
from its_project.features.economic_features import EconomicFeatures


@dataclass
class WalkForwardWindow:
    """Single walk-forward window configuration."""
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    window_id: int


@dataclass
class WalkForwardResult:
    """Results from a single walk-forward window."""
    window_id: int
    train_period: str
    test_period: str
    train_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    train_samples: int
    test_samples: int
    overfitting_score: float  # Higher means more overfitting


class WalkForwardValidator:
    """
    Walk-forward validation for time series models.
    
    Prevents look-ahead bias and tests temporal stability.
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        
        # Walk-forward parameters
        self.initial_train_size = config.get("initial_train_size", 0.6)  # 60% initial training
        self.test_size = config.get("test_size", 0.2)  # 20% testing
        self.step_size = config.get("step_size", 0.1)  # 10% step forward
        self.min_train_samples = config.get("min_train_samples", 1000)
        self.min_test_samples = config.get("min_test_samples", 200)
        
        # Validation parameters
        self.max_windows = config.get("max_windows", 10)
        self.performance_threshold = config.get("performance_threshold", 0.5)
        
        # Target and feature configuration
        self.target_config = config.get("target", {
            "horizon": 5,
            "threshold": 0.002,
            "target_type": "direction"
        })
        self.feature_config = config.get("features", {
            "return_periods": [1, 5, 15, 60],
            "volatility_windows": [5, 15, 60],
            "use_microstructure": True
        })
    
    def create_windows(self, data: pd.DataFrame) -> List[WalkForwardWindow]:
        """
        Create walk-forward windows from data.
        
        Args:
            data: Time series data indexed by datetime
            
        Returns:
            List of walk-forward windows
        """
        total_samples = len(data)
        windows = []
        
        # Calculate window sizes in samples
        train_samples = max(int(total_samples * self.initial_train_size), self.min_train_samples)
        test_samples = max(int(total_samples * self.test_size), self.min_test_samples)
        step_samples = max(int(total_samples * self.step_size), 100)
        
        window_id = 0
        
        # Create windows
        train_start_idx = 0
        
        while train_start_idx + train_samples + test_samples <= total_samples:
            train_end_idx = train_start_idx + train_samples
            test_start_idx = train_end_idx
            test_end_idx = test_start_idx + test_samples
            
            # Convert indices to datetime
            train_start = data.index[train_start_idx]
            train_end = data.index[train_end_idx - 1]
            test_start = data.index[test_start_idx]
            test_end = data.index[test_end_idx - 1]
            
            window = WalkForwardWindow(
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                window_id=window_id
            )
            
            windows.append(window)
            
            # Move forward
            train_start_idx += step_samples
            window_id += 1
            
            # Limit number of windows
            if window_id >= self.max_windows:
                break
        
        return windows
    
    def validate_window(self, data: pd.DataFrame, window: WalkForwardWindow,
                      model_class: type, model_config: Dict[str, Any],
                      backtester_class: type, backtester_config: Dict[str, Any]) -> WalkForwardResult:
        """
        Validate a single walk-forward window.
        
        Args:
            data: Full dataset
            window: Window configuration
            model_class: Model class to use
            model_config: Model configuration
            backtester_class: Backtester class to use
            backtester_config: Backtester configuration
            
        Returns:
            Window validation results
        """
        # Split data
        train_data = data.loc[window.train_start:window.train_end]
        test_data = data.loc[window.test_start:window.test_end]
        
        # Initialize components
        target_calculator = EconomicTargetCalculator(self.target_config)
        feature_calculator = EconomicFeatures(self.feature_config)
        
        # Calculate targets and features for training
        train_target, train_metadata = target_calculator.calculate_target(train_data)
        train_features = feature_calculator.calculate(train_data)
        
        # Align features and targets
        train_features, train_target = self._align_features_targets(train_features, train_target, train_data)
        
        # Train model
        model = model_class(model_config)
        model.fit(train_features, train_target)
        
        # Evaluate on training data
        train_metrics = self._evaluate_model(model, train_features, train_target)
        
        # Calculate targets and features for testing
        test_target, test_metadata = target_calculator.calculate_target(test_data)
        test_features = feature_calculator.calculate(test_data)
        
        # Align features and targets
        test_features, test_target = self._align_features_targets(test_features, test_target, test_data)
        
        # Evaluate on test data
        test_metrics = self._evaluate_model(model, test_features, test_target)
        
        # Calculate overfitting score
        overfitting_score = self._calculate_overfitting_score(train_metrics, test_metrics)
        
        return WalkForwardResult(
            window_id=window.window_id,
            train_period=f"{window.train_start.strftime('%Y-%m-%d')} to {window.train_end.strftime('%Y-%m-%d')}",
            test_period=f"{window.test_start.strftime('%Y-%m-%d')} to {window.test_end.strftime('%Y-%m-%d')}",
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            train_samples=len(train_features),
            test_samples=len(test_features),
            overfitting_score=overfitting_score
        )
    
    def _align_features_targets(self, features: np.ndarray, target: pd.Series, 
                              data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Align features and targets, removing NaN values."""
        # Convert to DataFrame if needed
        if isinstance(target, pd.Series):
            target_values = target.values
        else:
            target_values = target
        
        # Find valid indices (no NaN in either features or target)
        valid_indices = []
        
        for i in range(min(len(features), len(target_values))):
            if not np.isnan(target_values[i]) and not np.any(np.isnan(features[i])):
                valid_indices.append(i)
        
        if not valid_indices:
            raise ValueError("No valid samples after removing NaN values")
        
        # Align
        aligned_features = features[valid_indices]
        aligned_target = target_values[valid_indices]
        
        return aligned_features, aligned_target
    
    def _evaluate_model(self, model: BaseModel, features: np.ndarray, 
                       target: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance."""
        # Get predictions
        predictions = model.predict(features)
        probabilities = model.predict_proba(features)
        
        # Calculate metrics
        accuracy = np.mean(predictions == target)
        
        # Class-wise accuracy
        unique_classes = np.unique(target)
        class_accuracies = {}
        for cls in unique_classes:
            mask = target == cls
            if np.sum(mask) > 0:
                class_acc = np.mean(predictions[mask] == cls)
                class_accuracies[f"accuracy_class_{cls}"] = class_acc
        
        # F1 score (macro)
        from sklearn.metrics import f1_score, precision_score, recall_score
        f1_macro = f1_score(target, predictions, average='macro', zero_division=0)
        precision_macro = precision_score(target, predictions, average='macro', zero_division=0)
        recall_macro = recall_score(target, predictions, average='macro', zero_division=0)
        
        # Economic metrics
        confidences = model.get_confidence(features)
        avg_confidence = np.mean(confidences)
        
        # Trading signal quality
        buy_mask = (predictions == 2)
        sell_mask = (predictions == 0)
        
        buy_accuracy = np.mean(target[buy_mask] == 2) if np.sum(buy_mask) > 0 else 0
        sell_accuracy = np.mean(target[sell_mask] == 0) if np.sum(sell_mask) > 0 else 0
        
        metrics = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'avg_confidence': avg_confidence,
            'buy_signal_accuracy': buy_accuracy,
            'sell_signal_accuracy': sell_accuracy,
            'buy_signal_frequency': np.mean(buy_mask),
            'sell_signal_frequency': np.mean(sell_mask),
            **class_accuracies
        }
        
        return metrics
    
    def _calculate_overfitting_score(self, train_metrics: Dict[str, float], 
                                   test_metrics: Dict[str, float]) -> float:
        """Calculate overfitting score (0 = no overfitting, 1 = severe overfitting)."""
        overfitting_scores = []
        
        # Compare key metrics
        key_metrics = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
        
        for metric in key_metrics:
            if metric in train_metrics and metric in test_metrics:
                train_val = train_metrics[metric]
                test_val = test_metrics[metric]
                
                # Calculate relative performance drop
                if train_val > 0:
                    performance_drop = (train_val - test_val) / train_val
                    overfitting_scores.append(min(performance_drop, 1.0))
        
        # Average overfitting score
        if overfitting_scores:
            return np.mean(overfitting_scores)
        else:
            return 0.0
    
    def run_walk_forward_validation(self, data: pd.DataFrame, model_class: type,
                                 model_config: Dict[str, Any], backtester_class: type = None,
                                 backtester_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Run complete walk-forward validation.
        
        Args:
            data: Time series data
            model_class: Model class to validate
            model_config: Model configuration
            backtester_class: Optional backtester class
            backtester_config: Optional backtester configuration
            
        Returns:
            Complete validation results
        """
        # Create windows
        windows = self.create_windows(data)
        
        if not windows:
            raise ValueError("No valid walk-forward windows created")
        
        print(f"Created {len(windows)} walk-forward windows")
        
        # Validate each window
        results = []
        
        for window in windows:
            print(f"Validating window {window.window_id + 1}/{len(windows)}...")
            
            try:
                result = self.validate_window(
                    data, window, model_class, model_config,
                    backtester_class, backtester_config
                )
                results.append(result)
                
                print(f"  Train accuracy: {result.train_metrics['accuracy']:.3f}")
                print(f"  Test accuracy: {result.test_metrics['accuracy']:.3f}")
                print(f"  Overfitting score: {result.overfitting_score:.3f}")
                
            except Exception as e:
                print(f"  Error in window {window.window_id}: {e}")
                continue
        
        if not results:
            raise ValueError("No windows successfully validated")
        
        # Aggregate results
        aggregated_results = self._aggregate_results(results)
        
        return {
            'windows': results,
            'aggregated': aggregated_results,
            'validation_config': {
                'initial_train_size': self.initial_train_size,
                'test_size': self.test_size,
                'step_size': self.step_size,
                'total_windows': len(windows),
                'successful_windows': len(results)
            }
        }
    
    def _aggregate_results(self, results: List[WalkForwardResult]) -> Dict[str, Any]:
        """Aggregate results across all windows."""
        # Extract metrics
        train_accuracies = [r.train_metrics['accuracy'] for r in results]
        test_accuracies = [r.test_metrics['accuracy'] for r in results]
        overfitting_scores = [r.overfitting_score for r in results]
        
        # Calculate statistics
        aggregated = {
            'train_accuracy': {
                'mean': np.mean(train_accuracies),
                'std': np.std(train_accuracies),
                'min': np.min(train_accuracies),
                'max': np.max(train_accuracies)
            },
            'test_accuracy': {
                'mean': np.mean(test_accuracies),
                'std': np.std(test_accuracies),
                'min': np.min(test_accuracies),
                'max': np.max(test_accuracies)
            },
            'overfitting_score': {
                'mean': np.mean(overfitting_scores),
                'std': np.std(overfitting_scores),
                'min': np.min(overfitting_scores),
                'max': np.max(overfitting_scores)
            }
        }
        
        # Performance stability
        performance_stability = 1.0 - np.std(test_accuracies) / (np.mean(test_accuracies) + 1e-10)
        aggregated['performance_stability'] = performance_stability
        
        # Temporal degradation (compare first vs last window)
        if len(results) > 1:
            first_window_acc = results[0].test_metrics['accuracy']
            last_window_acc = results[-1].test_metrics['accuracy']
            temporal_degradation = (first_window_acc - last_window_acc) / first_window_acc
            aggregated['temporal_degradation'] = temporal_degradation
        
        # Success rate (windows meeting performance threshold)
        successful_windows = [r for r in results if r.test_metrics['accuracy'] >= self.performance_threshold]
        success_rate = len(successful_windows) / len(results)
        aggregated['success_rate'] = success_rate
        
        # Overall assessment
        mean_test_acc = np.mean(test_accuracies)
        mean_overfitting = np.mean(overfitting_scores)
        
        if mean_test_acc >= 0.6 and mean_overfitting <= 0.2:
            assessment = "EXCELLENT"
        elif mean_test_acc >= 0.55 and mean_overfitting <= 0.3:
            assessment = "GOOD"
        elif mean_test_acc >= 0.5 and mean_overfitting <= 0.4:
            assessment = "ACCEPTABLE"
        else:
            assessment = "POOR"
        
        aggregated['overall_assessment'] = assessment
        
        return aggregated
    
    def generate_validation_report(self, validation_results: Dict[str, Any]) -> str:
        """Generate comprehensive validation report."""
        windows = validation_results['windows']
        aggregated = validation_results['aggregated']
        config = validation_results['validation_config']
        
        report = []
        report.append("# WALK-FORWARD VALIDATION REPORT")
        report.append("")
        report.append(f"Total Windows: {config['total_windows']}")
        report.append(f"Successful Windows: {config['successful_windows']}")
        report.append(f"Success Rate: {aggregated['success_rate']:.2%}")
        report.append("")
        report.append("## PERFORMANCE SUMMARY")
        report.append("")
        report.append(f"Train Accuracy: {aggregated['train_accuracy']['mean']:.3f} ± {aggregated['train_accuracy']['std']:.3f}")
        report.append(f"Test Accuracy: {aggregated['test_accuracy']['mean']:.3f} ± {aggregated['test_accuracy']['std']:.3f}")
        report.append(f"Overfitting Score: {aggregated['overfitting_score']['mean']:.3f} ± {aggregated['overfitting_score']['std']:.3f}")
        report.append(f"Performance Stability: {aggregated['performance_stability']:.3f}")
        report.append("")
        report.append("## DETAILED RESULTS")
        report.append("")
        
        for result in windows:
            report.append(f"### Window {result.window_id + 1}")
            report.append(f"Train Period: {result.train_period}")
            report.append(f"Test Period: {result.test_period}")
            report.append(f"Train Accuracy: {result.train_metrics['accuracy']:.3f}")
            report.append(f"Test Accuracy: {result.test_metrics['accuracy']:.3f}")
            report.append(f"Overfitting Score: {result.overfitting_score:.3f}")
            report.append("")
        
        report.append("## ASSESSMENT")
        report.append(f"Overall Assessment: {aggregated['overall_assessment']}")
        
        if 'temporal_degradation' in aggregated:
            report.append(f"Temporal Degradation: {aggregated['temporal_degradation']:.3f}")
        
        return "\n".join(report)
