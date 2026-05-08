#!/usr/bin/env python3
"""
MLOps Drift Detection
=====================

Production-ready drift detection system:
- Data drift detection
- Model performance decay
- Concept drift monitoring
- Statistical tests
- Alerting system
- Automated retraining triggers
"""

from __future__ import annotations

import os
import json
import logging
import time
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import sqlite3
from enum import Enum
from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class DriftType(Enum):
    """Types of drift to detect."""
    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    PERFORMANCE_DRIFT = "performance_drift"
    LABEL_DRIFT = "label_drift"


class DriftSeverity(Enum):
    """Drift severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DriftDetectionResult:
    """Result of drift detection analysis."""
    test_name: str
    drift_type: DriftType
    detected: bool
    p_value: float
    statistic: float
    threshold: float
    severity: DriftSeverity
    timestamp: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DriftAlert:
    """Drift alert notification."""
    alert_id: str
    model_id: str
    drift_type: DriftType
    severity: DriftSeverity
    message: str
    timestamp: int
    metrics: Dict[str, float] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    resolved: bool = False
    resolution_time: Optional[int] = None


class DriftDetector:
    """
    Production-ready drift detection system.
    
    Features:
    - Multiple statistical tests
    - Performance monitoring
    - Concept drift detection
    - Alerting system
    - Automated retraining triggers
    """
    
    def __init__(
        self,
        base_path: str = "drift_detection",
        significance_level: float = 0.05,
        window_size: int = 1000,
        alert_thresholds: Optional[Dict[str, float]] = None
    ) -> None:
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Configuration
        self.significance_level = significance_level
        self.window_size = window_size
        self.alert_thresholds = alert_thresholds or {
            'performance_drop': 0.1,      # 10% performance drop
            'data_drift_p_value': 0.01,    # 1% significance
            'concept_drift_threshold': 0.15, # 15% concept drift
            'label_drift_ratio': 0.2       # 20% label distribution change
        }
        
        # Database
        self.db_path = self.base_path / "drift.db"
        self._init_database()
        
        # Reference data storage
        self.reference_data: Dict[str, np.ndarray] = {}
        self.reference_labels: Dict[str, np.ndarray] = {}
        self.reference_stats: Dict[str, Dict[str, float]] = {}
        
        # Current data buffer
        self.current_data_buffer: Dict[str, List[np.ndarray]] = {}
        self.current_labels_buffer: Dict[str, List[np.ndarray]] = {}
        
        # Performance history
        self.performance_history: Dict[str, List[Dict[str, float]]] = {}
        
        # Alerts
        self.active_alerts: Dict[str, DriftAlert] = {}
        self.alert_callbacks: List[callable] = []
        
        # Statistics
        self.stats = {
            'total_detections': 0,
            'data_drifts': 0,
            'concept_drifts': 0,
            'performance_drifts': 0,
            'label_drifts': 0,
            'alerts_triggered': 0,
            'auto_retrains_triggered': 0
        }
    
    def _init_database(self) -> None:
        """Initialize SQLite database for drift tracking."""
        with sqlite3.connect(self.db_path) as conn:
            # Drift results table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS drift_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT,
                    test_name TEXT,
                    drift_type TEXT,
                    detected BOOLEAN,
                    p_value REAL,
                    statistic REAL,
                    threshold REAL,
                    severity TEXT,
                    timestamp INTEGER,
                    metadata TEXT
                )
            """)
            
            # Alerts table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS drift_alerts (
                    alert_id TEXT PRIMARY KEY,
                    model_id TEXT,
                    drift_type TEXT,
                    severity TEXT,
                    message TEXT,
                    timestamp INTEGER,
                    metrics TEXT,
                    recommendations TEXT,
                    resolved BOOLEAN,
                    resolution_time INTEGER
                )
            """)
            
            # Indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_drift_model ON drift_results(model_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_drift_timestamp ON drift_results(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_model ON drift_alerts(model_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON drift_alerts(timestamp)")
    
    def set_reference_data(
        self,
        model_id: str,
        features: np.ndarray,
        labels: Optional[np.ndarray] = None
    ) -> None:
        """
        Set reference data for drift detection.
        
        Args:
            model_id: Model identifier
            features: Reference feature data
            labels: Reference labels (optional)
        """
        self.reference_data[model_id] = features
        if labels is not None:
            self.reference_labels[model_id] = labels
        
        # Calculate reference statistics
        self.reference_stats[model_id] = {
            'mean': np.mean(features, axis=0),
            'std': np.std(features, axis=0),
            'min': np.min(features, axis=0),
            'max': np.max(features, axis=0),
            'median': np.median(features, axis=0),
            'q25': np.percentile(features, 25, axis=0),
            'q75': np.percentile(features, 75, axis=0)
        }
        
        # Initialize buffers
        self.current_data_buffer[model_id] = []
        self.current_labels_buffer[model_id] = []
        self.performance_history[model_id] = []
        
        logger.info(f"Reference data set for model {model_id}: {features.shape}")
    
    def add_current_data(
        self,
        model_id: str,
        features: np.ndarray,
        labels: Optional[np.ndarray] = None,
        predictions: Optional[np.ndarray] = None,
        actuals: Optional[np.ndarray] = None
    ) -> List[DriftDetectionResult]:
        """
        Add current data and check for drift.
        
        Args:
            model_id: Model identifier
            features: Current feature data
            labels: Current labels (optional)
            predictions: Model predictions (optional)
            actuals: Actual values (optional)
            
        Returns:
            List of drift detection results
        """
        if model_id not in self.reference_data:
            logger.warning(f"No reference data for model {model_id}")
            return []
        
        # Add to buffer
        self.current_data_buffer[model_id].append(features)
        if labels is not None:
            self.current_labels_buffer[model_id].append(labels)
        
        # Check if we have enough data
        buffer = self.current_data_buffer[model_id]
        total_samples = sum(arr.shape[0] for arr in buffer)
        
        if total_samples < self.window_size:
            return []
        
        # Combine buffer data
        current_features = np.vstack(buffer)
        current_labels = np.vstack(self.current_labels_buffer[model_id]) if self.current_labels_buffer[model_id] else None
        
        # Keep only recent data
        if len(buffer) > 10:  # Keep last 10 batches
            self.current_data_buffer[model_id] = buffer[-10:]
            if self.current_labels_buffer[model_id]:
                self.current_labels_buffer[model_id] = self.current_labels_buffer[model_id][-10:]
        
        # Run drift detection
        results = []
        
        # Data drift detection
        data_drift_results = self._detect_data_drift(model_id, current_features)
        results.extend(data_drift_results)
        
        # Label drift detection
        if current_labels is not None and model_id in self.reference_labels:
            label_drift_results = self._detect_label_drift(model_id, current_labels)
            results.extend(label_drift_results)
        
        # Performance drift detection
        if predictions is not None and actuals is not None:
            perf_drift_results = self._detect_performance_drift(model_id, predictions, actuals)
            results.extend(perf_drift_results)
        
        # Concept drift detection
        if predictions is not None and current_labels is not None:
            concept_drift_results = self._detect_concept_drift(model_id, current_features, predictions, current_labels)
            results.extend(concept_drift_results)
        
        # Save results
        self._save_drift_results(model_id, results)
        
        # Update statistics
        for result in results:
            if result.detected:
                self.stats['total_detections'] += 1
                if result.drift_type == DriftType.DATA_DRIFT:
                    self.stats['data_drifts'] += 1
                elif result.drift_type == DriftType.CONCEPT_DRIFT:
                    self.stats['concept_drifts'] += 1
                elif result.drift_type == DriftType.PERFORMANCE_DRIFT:
                    self.stats['performance_drifts'] += 1
                elif result.drift_type == DriftType.LABEL_DRIFT:
                    self.stats['label_drifts'] += 1
        
        return results
    
    def _detect_data_drift(self, model_id: str, current_features: np.ndarray) -> List[DriftDetectionResult]:
        """Detect data distribution drift."""
        if model_id not in self.reference_data:
            return []
        
        reference = self.reference_data[model_id]
        results = []
        
        # Kolmogorov-Smirnov test for each feature
        for i in range(min(reference.shape[1], current_features.shape[1])):
            ref_feature = reference[:, i]
            curr_feature = current_features[:, i]
            
            # KS test
            ks_stat, ks_p_value = stats.ks_2samp(ref_feature, curr_feature)
            
            detected = ks_p_value < self.alert_thresholds['data_drift_p_value']
            severity = self._calculate_severity(ks_p_value, self.alert_thresholds['data_drift_p_value'])
            
            result = DriftDetectionResult(
                test_name=f"ks_test_feature_{i}",
                drift_type=DriftType.DATA_DRIFT,
                detected=detected,
                p_value=ks_p_value,
                statistic=ks_stat,
                threshold=self.alert_thresholds['data_drift_p_value'],
                severity=severity,
                timestamp=int(time.time() * 1000),
                metadata={
                    'feature_index': i,
                    'reference_mean': np.mean(ref_feature),
                    'current_mean': np.mean(curr_feature),
                    'reference_std': np.std(ref_feature),
                    'current_std': np.std(curr_feature)
                }
            )
            results.append(result)
        
        # Population Stability Index (PSI)
        psi_result = self._calculate_psi(model_id, current_features)
        if psi_result:
            results.append(psi_result)
        
        return results
    
    def _detect_label_drift(self, model_id: str, current_labels: np.ndarray) -> List[DriftDetectionResult]:
        """Detect label distribution drift."""
        if model_id not in self.reference_labels:
            return []
        
        reference = self.reference_labels[model_id]
        results = []
        
        # Chi-square test for categorical labels
        if len(reference.shape) == 1 or reference.shape[1] == 1:
            # Binary/multiclass classification
            ref_labels = reference.flatten()
            curr_labels = current_labels.flatten()
            
            # Create frequency tables
            unique_labels = np.unique(np.concatenate([ref_labels, curr_labels]))
            ref_counts = [np.sum(ref_labels == label) for label in unique_labels]
            curr_counts = [np.sum(curr_labels == label) for label in unique_labels]
            
            # Chi-square test
            chi2_stat, chi2_p_value = stats.chisquare(curr_counts, ref_counts)
            
            detected = chi2_p_value < self.significance_level
            severity = self._calculate_severity(chi2_p_value, self.significance_level)
            
            result = DriftDetectionResult(
                test_name="chi2_label_distribution",
                drift_type=DriftType.LABEL_DRIFT,
                detected=detected,
                p_value=chi2_p_value,
                statistic=chi2_stat,
                threshold=self.significance_level,
                severity=severity,
                timestamp=int(time.time() * 1000),
                metadata={
                    'reference_distribution': dict(zip(unique_labels, ref_counts)),
                    'current_distribution': dict(zip(unique_labels, curr_counts)),
                    'total_reference': len(ref_labels),
                    'total_current': len(curr_labels)
                }
            )
            results.append(result)
        
        return results
    
    def _detect_performance_drift(
        self,
        model_id: str,
        predictions: np.ndarray,
        actuals: np.ndarray
    ) -> List[DriftDetectionResult]:
        """Detect model performance decay."""
        results = []
        
        # Calculate current performance
        current_metrics = self._calculate_performance_metrics(predictions, actuals)
        
        # Add to history
        self.performance_history[model_id].append({
            'timestamp': int(time.time() * 1000),
            **current_metrics
        })
        
        # Keep only recent history
        if len(self.performance_history[model_id]) > 100:
            self.performance_history[model_id] = self.performance_history[model_id][-100:]
        
        # Compare with baseline
        if len(self.performance_history[model_id]) >= 10:
            # Use first 10 measurements as baseline
            baseline_metrics = self.performance_history[model_id][:10]
            
            for metric_name, current_value in current_metrics.items():
                baseline_values = [m[metric_name] for m in baseline_metrics if metric_name in m]
                if not baseline_values:
                    continue
                
                baseline_mean = np.mean(baseline_values)
                drop_ratio = (baseline_mean - current_value) / baseline_mean
                
                detected = drop_ratio > self.alert_thresholds['performance_drop']
                severity = self._calculate_severity(drop_ratio, self.alert_thresholds['performance_drop'])
                
                result = DriftDetectionResult(
                    test_name=f"performance_{metric_name}",
                    drift_type=DriftType.PERFORMANCE_DRIFT,
                    detected=detected,
                    p_value=drop_ratio,
                    statistic=current_value,
                    threshold=self.alert_thresholds['performance_drop'],
                    severity=severity,
                    timestamp=int(time.time() * 1000),
                    metadata={
                        'metric_name': metric_name,
                        'baseline_mean': baseline_mean,
                        'current_value': current_value,
                        'drop_ratio': drop_ratio,
                        'baseline_samples': len(baseline_values)
                    }
                )
                results.append(result)
        
        return results
    
    def _detect_concept_drift(
        self,
        model_id: str,
        features: np.ndarray,
        predictions: np.ndarray,
        labels: np.ndarray
    ) -> List[DriftDetectionResult]:
        """Detect concept drift (relationship between features and labels)."""
        results = []
        
        # Calculate feature-label correlation drift
        if model_id in self.reference_data and model_id in self.reference_labels:
            ref_features = self.reference_data[model_id]
            ref_labels = self.reference_labels[model_id]
            
            # Calculate correlations for reference
            ref_correlations = []
            for i in range(min(ref_features.shape[1], features.shape[1])):
                corr = np.corrcoef(ref_features[:, i], ref_labels.flatten())[0, 1]
                if not np.isnan(corr):
                    ref_correlations.append(abs(corr))
            
            # Calculate correlations for current
            curr_correlations = []
            for i in range(min(features.shape[1], labels.shape[1])):
                corr = np.corrcoef(features[:, i], labels.flatten())[0, 1]
                if not np.isnan(corr):
                    curr_correlations.append(abs(corr))
            
            if ref_correlations and curr_correlations:
                ref_mean_corr = np.mean(ref_correlations)
                curr_mean_corr = np.mean(curr_correlations)
                
                correlation_change = abs(ref_mean_corr - curr_mean_corr)
                detected = correlation_change > self.alert_thresholds['concept_drift_threshold']
                severity = self._calculate_severity(correlation_change, self.alert_thresholds['concept_drift_threshold'])
                
                result = DriftDetectionResult(
                    test_name="correlation_drift",
                    drift_type=DriftType.CONCEPT_DRIFT,
                    detected=detected,
                    p_value=correlation_change,
                    statistic=curr_mean_corr,
                    threshold=self.alert_thresholds['concept_drift_threshold'],
                    severity=severity,
                    timestamp=int(time.time() * 1000),
                    metadata={
                        'reference_mean_correlation': ref_mean_corr,
                        'current_mean_correlation': curr_mean_corr,
                        'correlation_change': correlation_change,
                        'reference_features': len(ref_correlations),
                        'current_features': len(curr_correlations)
                    }
                )
                results.append(result)
        
        return results
    
    def _calculate_psi(self, model_id: str, current_features: np.ndarray) -> Optional[DriftDetectionResult]:
        """Calculate Population Stability Index."""
        if model_id not in self.reference_stats:
            return None
        
        ref_stats = self.reference_stats[model_id]
        psi_values = []
        
        for i in range(min(len(ref_stats['mean']), current_features.shape[1])):
            ref_feature = self.reference_data[model_id][:, i]
            curr_feature = current_features[:, i]
            
            # Create bins
            all_data = np.concatenate([ref_feature, curr_feature])
            bins = np.percentile(all_data, [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
            
            # Calculate frequencies
            ref_hist, _ = np.histogram(ref_feature, bins=bins)
            curr_hist, _ = np.histogram(curr_feature, bins=bins)
            
            # Avoid division by zero
            ref_hist = ref_hist + 1e-10
            curr_hist = curr_hist + 1e-10
            
            # Calculate PSI
            ref_pct = ref_hist / len(ref_feature)
            curr_pct = curr_hist / len(curr_feature)
            
            psi = np.sum((ref_pct - curr_pct) * np.log(ref_pct / curr_pct))
            psi_values.append(psi)
        
        if psi_values:
            avg_psi = np.mean(psi_values)
            
            # PSI thresholds
            if avg_psi < 0.1:
                severity = DriftSeverity.LOW
            elif avg_psi < 0.25:
                severity = DriftSeverity.MEDIUM
            else:
                severity = DriftSeverity.HIGH
            
            detected = avg_psi > 0.1  # PSI > 0.1 indicates drift
            
            return DriftDetectionResult(
                test_name="population_stability_index",
                drift_type=DriftType.DATA_DRIFT,
                detected=detected,
                p_value=avg_psi,
                statistic=avg_psi,
                threshold=0.1,
                severity=severity,
                timestamp=int(time.time() * 1000),
                metadata={
                    'psi_values': psi_values,
                    'average_psi': avg_psi,
                    'features_analyzed': len(psi_values)
                }
            )
        
        return None
    
    def _calculate_performance_metrics(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray
    ) -> Dict[str, float]:
        """Calculate performance metrics."""
        metrics = {}
        
        # Classification metrics
        if len(predictions.shape) == 1 or predictions.shape[1] == 1:
            # Binary/multiclass classification
            pred_labels = (predictions.flatten() > 0.5).astype(int)
            true_labels = actuals.flatten().astype(int)
            
            try:
                metrics['accuracy'] = accuracy_score(true_labels, pred_labels)
                metrics['precision'] = precision_score(true_labels, pred_labels, average='weighted', zero_division=0)
                metrics['recall'] = recall_score(true_labels, pred_labels, average='weighted', zero_division=0)
                metrics['f1_score'] = f1_score(true_labels, pred_labels, average='weighted', zero_division=0)
            except:
                pass
        
        # Regression metrics
        else:
            try:
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                
                metrics['mse'] = mean_squared_error(actuals, predictions)
                metrics['mae'] = mean_absolute_error(actuals, predictions)
                metrics['r2_score'] = r2_score(actuals, predictions)
            except:
                pass
        
        return metrics
    
    def _calculate_severity(self, value: float, threshold: float) -> DriftSeverity:
        """Calculate drift severity based on value and threshold."""
        ratio = value / threshold
        
        if ratio < 1.5:
            return DriftSeverity.LOW
        elif ratio < 2.0:
            return DriftSeverity.MEDIUM
        elif ratio < 3.0:
            return DriftSeverity.HIGH
        else:
            return DriftSeverity.CRITICAL
    
    def _save_drift_results(self, model_id: str, results: List[DriftDetectionResult]) -> None:
        """Save drift detection results to database."""
        with sqlite3.connect(self.db_path) as conn:
            for result in results:
                conn.execute("""
                    INSERT INTO drift_results VALUES (
                        NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                """, (
                    model_id, result.test_name, result.drift_type.value,
                    result.detected, result.p_value, result.statistic,
                    result.threshold, result.severity.value, result.timestamp,
                    json.dumps(result.metadata)
                ))
        
        # Trigger alerts for detected drift
        for result in results:
            if result.detected:
                self._trigger_alert(model_id, result)
    
    def _trigger_alert(self, model_id: str, result: DriftDetectionResult) -> None:
        """Trigger drift alert."""
        alert_id = f"{model_id}_{result.drift_type.value}_{int(time.time())}"
        
        # Create alert message
        message = f"{result.drift_type.value.replace('_', ' ').title()} detected in model {model_id}"
        if result.drift_type == DriftType.DATA_DRIFT:
            message += f" (p-value: {result.p_value:.4f})"
        elif result.drift_type == DriftType.PERFORMANCE_DRIFT:
            message += f" (drop: {result.p_value:.2%})"
        
        # Generate recommendations
        recommendations = self._generate_recommendations(result)
        
        alert = DriftAlert(
            alert_id=alert_id,
            model_id=model_id,
            drift_type=result.drift_type,
            severity=result.severity,
            message=message,
            timestamp=result.timestamp,
            metrics={
                'p_value': result.p_value,
                'statistic': result.statistic,
                'threshold': result.threshold
            },
            recommendations=recommendations
        )
        
        # Save alert
        self.active_alerts[alert_id] = alert
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO drift_alerts VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            """, (
                alert.alert_id, alert.model_id, alert.drift_type.value,
                alert.severity.value, alert.message, alert.timestamp,
                json.dumps(alert.metrics), json.dumps(alert.recommendations),
                alert.resolved, alert.resolution_time
            ))
        
        # Update statistics
        self.stats['alerts_triggered'] += 1
        
        # Trigger callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
        
        # Check for auto-retrain trigger
        if result.severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL]:
            self._trigger_auto_retrain(model_id, result)
        
        logger.warning(f"Drift alert triggered: {message}")
    
    def _generate_recommendations(self, result: DriftDetectionResult) -> List[str]:
        """Generate recommendations for drift mitigation."""
        recommendations = []
        
        if result.drift_type == DriftType.DATA_DRIFT:
            recommendations.extend([
                "Collect more recent training data",
                "Update data preprocessing pipeline",
                "Consider feature engineering adjustments",
                "Monitor data source quality"
            ])
        elif result.drift_type == DriftType.CONCEPT_DRIFT:
            recommendations.extend([
                "Retrain model with recent data",
                "Consider online learning approach",
                "Update feature selection",
                "Review model architecture"
            ])
        elif result.drift_type == DriftType.PERFORMANCE_DRIFT:
            recommendations.extend([
                "Retrain model immediately",
                "Check for data quality issues",
                "Validate model assumptions",
                "Consider ensemble methods"
            ])
        elif result.drift_type == DriftType.LABEL_DRIFT:
            recommendations.extend([
                "Update label distribution in training",
                "Consider class imbalance handling",
                "Review labeling process",
                "Collect more balanced data"
            ])
        
        if result.severity == DriftSeverity.CRITICAL:
            recommendations.insert(0, "IMMEDIATE ACTION REQUIRED - Consider model rollback")
        
        return recommendations
    
    def _trigger_auto_retrain(self, model_id: str, result: DriftDetectionResult) -> None:
        """Trigger automatic retraining."""
        self.stats['auto_retrains_triggered'] += 1
        logger.critical(f"Auto-retrain triggered for model {model_id} due to {result.drift_type.value}")
        
        # This would integrate with the training pipeline
        # For now, just log the event
        auto_retrain_event = {
            'model_id': model_id,
            'trigger_reason': result.drift_type.value,
            'severity': result.severity.value,
            'timestamp': int(time.time() * 1000),
            'recommendations': self._generate_recommendations(result)
        }
        
        # Save auto-retrain event
        with open(self.base_path / f"auto_retrain_{model_id}_{int(time.time())}.json", 'w') as f:
            json.dump(auto_retrain_event, f, indent=2)
    
    def add_alert_callback(self, callback: callable) -> None:
        """Add callback for drift alerts."""
        self.alert_callbacks.append(callback)
    
    def get_drift_history(
        self,
        model_id: Optional[str] = None,
        drift_type: Optional[DriftType] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get drift detection history."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = "SELECT * FROM drift_results WHERE 1=1"
                params = []
                
                if model_id:
                    query += " AND model_id = ?"
                    params.append(model_id)
                
                if drift_type:
                    query += " AND drift_type = ?"
                    params.append(drift_type.value)
                
                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)
                
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()
                
                return [
                    {
                        'id': row[0],
                        'model_id': row[1],
                        'test_name': row[2],
                        'drift_type': row[3],
                        'detected': bool(row[4]),
                        'p_value': row[5],
                        'statistic': row[6],
                        'threshold': row[7],
                        'severity': row[8],
                        'timestamp': row[9],
                        'metadata': json.loads(row[10]) if row[10] else {}
                    }
                    for row in rows
                ]
                
        except Exception as e:
            logger.error(f"Error getting drift history: {e}")
            return []
    
    def get_active_alerts(self, model_id: Optional[str] = None) -> List[DriftAlert]:
        """Get active drift alerts."""
        alerts = []
        
        for alert in self.active_alerts.values():
            if model_id is None or alert.model_id == model_id:
                alerts.append(alert)
        
        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve a drift alert."""
        if alert_id not in self.active_alerts:
            return False
        
        alert = self.active_alerts[alert_id]
        alert.resolved = True
        alert.resolution_time = int(time.time() * 1000)
        
        # Update database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE drift_alerts SET resolved = 1, resolution_time = ?
                WHERE alert_id = ?
            """, (alert.resolution_time, alert_id))
        
        logger.info(f"Alert resolved: {alert_id}")
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get drift detection statistics."""
        base_stats = self.stats.copy()
        
        # Add current status
        base_stats.update({
            'models_monitored': len(self.reference_data),
            'active_alerts': len([a for a in self.active_alerts.values() if not a.resolved]),
            'total_alerts': len(self.active_alerts),
            'significance_level': self.significance_level,
            'window_size': self.window_size,
            'alert_thresholds': self.alert_thresholds
        })
        
        return base_stats


# Convenience functions
def create_drift_detector(
    base_path: str = "drift_detection",
    significance_level: float = 0.05,
    window_size: int = 1000
) -> DriftDetector:
    """Create drift detector with default settings."""
    return DriftDetector(
        base_path=base_path,
        significance_level=significance_level,
        window_size=window_size
    )


def create_conservative_drift_detector() -> DriftDetector:
    """Create conservative drift detector (more sensitive)."""
    alert_thresholds = {
        'performance_drop': 0.05,      # 5% performance drop
        'data_drift_p_value': 0.05,    # 5% significance
        'concept_drift_threshold': 0.10, # 10% concept drift
        'label_drift_ratio': 0.15       # 15% label distribution change
    }
    
    return DriftDetector(
        significance_level=0.1,
        window_size=500,
        alert_thresholds=alert_thresholds
    )


if __name__ == "__main__":
    # Test drift detector
    logging.basicConfig(level=logging.INFO)
    
    detector = create_drift_detector("test_drift")
    
    # Create synthetic reference data
    np.random.seed(42)
    n_samples = 1000
    n_features = 5
    
    reference_features = np.random.randn(n_samples, n_features)
    reference_labels = (reference_features[:, 0] > 0).astype(int)
    
    # Set reference data
    detector.set_reference_data("test_model", reference_features, reference_labels)
    
    print(f"Reference data set: {reference_features.shape}")
    
    # Test with no drift (similar distribution)
    current_features_1 = np.random.randn(100, n_features)
    current_labels_1 = (current_features_1[:, 0] > 0).astype(int)
    
    results_1 = detector.add_current_data(
        "test_model",
        current_features_1,
        current_labels_1
    )
    
    print(f"Detection results (no drift): {len(results_1)} alerts")
    
    # Test with data drift (different distribution)
    current_features_2 = np.random.randn(100, n_features) + 2.0  # Shifted distribution
    current_labels_2 = (current_features_2[:, 0] > 0).astype(int)
    
    results_2 = detector.add_current_data(
        "test_model",
        current_features_2,
        current_labels_2
    )
    
    print(f"Detection results (with drift): {len(results_2)} alerts")
    for result in results_2:
        if result.detected:
            print(f"  {result.drift_type.value}: {result.test_name} (p={result.p_value:.4f})")
    
    # Get statistics
    stats = detector.get_statistics()
    print(f"\nDrift detection statistics:")
    print(f"  Total detections: {stats['total_detections']}")
    print(f"  Data drifts: {stats['data_drifts']}")
    print(f"  Alerts triggered: {stats['alerts_triggered']}")
    print(f"  Models monitored: {stats['models_monitored']}")
