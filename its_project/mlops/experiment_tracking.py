#!/usr/bin/env python3
"""
MLOps Experiment Tracking
========================

Production-ready experiment tracking system (MLflow alternative):
- Experiment management
- Parameter tracking
- Metrics logging
- Artifact storage
- Model comparison
- Hyperparameter optimization
"""

from __future__ import annotations

import os
import json
import logging
import time
import uuid
import pickle
import shutil
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
import sqlite3
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class ExperimentStatus(Enum):
    """Experiment status types."""
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SCHEDULED = "scheduled"
    CANCELLED = "cancelled"


@dataclass
class ExperimentRun:
    """Single experiment run metadata."""
    run_id: str
    experiment_id: str
    name: str
    status: ExperimentStatus
    start_time: int
    end_time: Optional[int] = None
    duration_ms: Optional[int] = None
    
    # Parameters
    parameters: Dict[str, Any] = field(default_factory=dict)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    
    # Metrics
    metrics: Dict[str, float] = field(default_factory=dict)
    training_metrics: Dict[str, float] = field(default_factory=dict)
    validation_metrics: Dict[str, float] = field(default_factory=dict)
    test_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Trading-specific metrics
    trading_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Artifacts
    model_path: Optional[str] = None
    artifacts: Dict[str, str] = field(default_factory=dict)
    
    # System info
    git_commit: Optional[str] = None
    environment: Dict[str, Any] = field(default_factory=dict)
    
    # Error handling
    error: Optional[str] = None
    traceback: Optional[str] = None


@dataclass
class Experiment:
    """Experiment metadata."""
    experiment_id: str
    name: str
    description: str
    created_at: int
    created_by: str
    tags: List[str] = field(default_factory=list)
    
    # Best run tracking
    best_run_id: Optional[str] = None
    best_metric: Optional[str] = None
    best_value: Optional[float] = None
    
    # Configuration
    metric_direction: str = "maximize"  # "maximize" or "minimize"
    primary_metric: str = "sharpe_ratio"


class ExperimentTracker:
    """
    Production-ready experiment tracking system.
    
    Features:
    - Experiment and run management
    - Parameter and metrics tracking
    - Artifact storage
    - Model comparison
    - Visualization
    - Search and filtering
    """
    
    def __init__(self, base_path: str = "mlflow_registry") -> None:
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Directory structure
        self.experiments_dir = self.base_path / "experiments"
        self.runs_dir = self.base_path / "runs"
        self.artifacts_dir = self.base_path / "artifacts"
        self.models_dir = self.base_path / "models"
        self.plots_dir = self.base_path / "plots"
        self.db_path = self.base_path / "tracking.db"
        
        # Create directories
        for directory in [self.experiments_dir, self.runs_dir, self.artifacts_dir, 
                        self.models_dir, self.plots_dir]:
            directory.mkdir(exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # Active runs tracking
        self.active_runs: Dict[str, ExperimentRun] = {}
        
        # Configuration
        self.auto_save_interval = 60  # seconds
    
    def _init_database(self) -> None:
        """Initialize SQLite database for experiment tracking."""
        with sqlite3.connect(self.db_path) as conn:
            # Experiments table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    name TEXT,
                    description TEXT,
                    created_at INTEGER,
                    created_by TEXT,
                    tags TEXT,
                    best_run_id TEXT,
                    best_metric TEXT,
                    best_value REAL,
                    metric_direction TEXT,
                    primary_metric TEXT
                )
            """)
            
            # Runs table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    experiment_id TEXT,
                    name TEXT,
                    status TEXT,
                    start_time INTEGER,
                    end_time INTEGER,
                    duration_ms INTEGER,
                    parameters TEXT,
                    hyperparameters TEXT,
                    metrics TEXT,
                    training_metrics TEXT,
                    validation_metrics TEXT,
                    test_metrics TEXT,
                    trading_metrics TEXT,
                    model_path TEXT,
                    artifacts TEXT,
                    git_commit TEXT,
                    environment TEXT,
                    error TEXT,
                    traceback TEXT,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id)
                )
            """)
            
            # Indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_experiment ON runs(experiment_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_start_time ON runs(start_time)")
    
    def create_experiment(
        self,
        name: str,
        description: str = "",
        tags: Optional[List[str]] = None,
        created_by: str = "system",
        primary_metric: str = "sharpe_ratio",
        metric_direction: str = "maximize"
    ) -> str:
        """
        Create a new experiment.
        
        Args:
            name: Experiment name
            description: Experiment description
            tags: List of tags
            created_by: Creator name
            primary_metric: Primary metric for comparison
            metric_direction: "maximize" or "minimize"
            
        Returns:
            Experiment ID
        """
        experiment_id = str(uuid.uuid4())
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO experiments VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            """, (
                experiment_id, name, description, int(time.time() * 1000),
                created_by, json.dumps(tags or []), None, primary_metric,
                None, metric_direction, primary_metric
            ))
        
        logger.info(f"Experiment created: {name} ({experiment_id})")
        return experiment_id
    
    def start_run(
        self,
        experiment_id: str,
        run_name: str,
        parameters: Optional[Dict[str, Any]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Start a new experiment run.
        
        Args:
            experiment_id: Experiment ID
            run_name: Run name
            parameters: Fixed parameters
            hyperparameters: Tunable hyperparameters
            tags: Run tags
            
        Returns:
            Run ID
        """
        run_id = str(uuid.uuid4())
        start_time = int(time.time() * 1000)
        
        # Get environment info
        env_info = self._get_environment_info()
        
        run = ExperimentRun(
            run_id=run_id,
            experiment_id=experiment_id,
            name=run_name,
            status=ExperimentStatus.RUNNING,
            start_time=start_time,
            parameters=parameters or {},
            hyperparameters=hyperparameters or {},
            environment=env_info
        )
        
        # Save to database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO runs VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            """, (
                run.run_id, run.experiment_id, run.name, run.status.value,
                run.start_time, run.end_time, run.duration_ms,
                json.dumps(run.parameters), json.dumps(run.hyperparameters),
                json.dumps(run.metrics), json.dumps(run.training_metrics),
                json.dumps(run.validation_metrics), json.dumps(run.test_metrics),
                json.dumps(run.trading_metrics), run.model_path,
                json.dumps(run.artifacts), run.git_commit,
                json.dumps(run.environment), run.error, run.traceback
            ))
        
        # Track active run
        self.active_runs[run_id] = run
        
        # Create run directory
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(exist_ok=True)
        
        # Save run config
        config = {
            'run_id': run_id,
            'experiment_id': experiment_id,
            'name': run_name,
            'parameters': parameters,
            'hyperparameters': hyperparameters,
            'tags': tags,
            'start_time': start_time
        }
        
        with open(run_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Run started: {run_name} ({run_id})")
        return run_id
    
    def log_metric(
        self,
        run_id: str,
        key: str,
        value: float,
        step: Optional[int] = None,
        timestamp: Optional[int] = None
    ) -> None:
        """
        Log a metric for a run.
        
        Args:
            run_id: Run ID
            key: Metric name
            value: Metric value
            step: Step number (optional)
            timestamp: Timestamp (optional)
        """
        if run_id not in self.active_runs:
            logger.warning(f"Run {run_id} not found in active runs")
            return
        
        # Update in-memory run
        run = self.active_runs[run_id]
        run.metrics[key] = value
        
        # Save to database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE runs SET metrics = ? WHERE run_id = ?
            """, (json.dumps(run.metrics), run_id))
        
        # Log to file for time series
        if step is not None:
            self._log_metric_timeseries(run_id, key, value, step, timestamp)
        
        logger.debug(f"Metric logged: {key} = {value} for run {run_id}")
    
    def log_metrics(
        self,
        run_id: str,
        metrics: Dict[str, float],
        category: str = "metrics"
    ) -> None:
        """
        Log multiple metrics for a run.
        
        Args:
            run_id: Run ID
            metrics: Dictionary of metrics
            category: Metric category (metrics, training, validation, test, trading)
        """
        if run_id not in self.active_runs:
            logger.warning(f"Run {run_id} not found in active runs")
            return
        
        run = self.active_runs[run_id]
        
        if category == "training":
            run.training_metrics.update(metrics)
        elif category == "validation":
            run.validation_metrics.update(metrics)
        elif category == "test":
            run.test_metrics.update(metrics)
        elif category == "trading":
            run.trading_metrics.update(metrics)
        else:
            run.metrics.update(metrics)
        
        # Save to database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE runs SET 
                    metrics = ?, training_metrics = ?, validation_metrics = ?,
                    test_metrics = ?, trading_metrics = ?
                WHERE run_id = ?
            """, (
                json.dumps(run.metrics), json.dumps(run.training_metrics),
                json.dumps(run.validation_metrics), json.dumps(run.test_metrics),
                json.dumps(run.trading_metrics), run_id
            ))
        
        logger.debug(f"Metrics logged for run {run_id}: {category}")
    
    def log_parameter(
        self,
        run_id: str,
        key: str,
        value: Any
    ) -> None:
        """
        Log a parameter for a run.
        
        Args:
            run_id: Run ID
            key: Parameter name
            value: Parameter value
        """
        if run_id not in self.active_runs:
            logger.warning(f"Run {run_id} not found in active runs")
            return
        
        run = self.active_runs[run_id]
        run.parameters[key] = value
        
        # Save to database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE runs SET parameters = ? WHERE run_id = ?
            """, (json.dumps(run.parameters), run_id))
        
        logger.debug(f"Parameter logged: {key} = {value} for run {run_id}")
    
    def log_artifact(
        self,
        run_id: str,
        artifact_path: str,
        artifact_name: Optional[str] = None
    ) -> str:
        """
        Log an artifact for a run.
        
        Args:
            run_id: Run ID
            artifact_path: Path to artifact file
            artifact_name: Artifact name (optional)
            
        Returns:
            Path to stored artifact
        """
        if run_id not in self.active_runs:
            logger.warning(f"Run {run_id} not found in active runs")
            return ""
        
        if not os.path.exists(artifact_path):
            logger.error(f"Artifact file not found: {artifact_path}")
            return ""
        
        # Create artifacts directory for run
        run_artifacts_dir = self.runs_dir / run_id / "artifacts"
        run_artifacts_dir.mkdir(exist_ok=True)
        
        # Copy artifact
        if artifact_name is None:
            artifact_name = os.path.basename(artifact_path)
        
        stored_path = run_artifacts_dir / artifact_name
        shutil.copy2(artifact_path, stored_path)
        
        # Update run metadata
        run = self.active_runs[run_id]
        run.artifacts[artifact_name] = str(stored_path)
        
        # Save to database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE runs SET artifacts = ? WHERE run_id = ?
            """, (json.dumps(run.artifacts), run_id))
        
        logger.info(f"Artifact logged: {artifact_name} for run {run_id}")
        return str(stored_path)
    
    def log_model(
        self,
        run_id: str,
        model_path: str,
        model_name: Optional[str] = None
    ) -> str:
        """
        Log a model for a run.
        
        Args:
            run_id: Run ID
            model_path: Path to model file
            model_name: Model name (optional)
            
        Returns:
            Path to stored model
        """
        if run_id not in self.active_runs:
            logger.warning(f"Run {run_id} not found in active runs")
            return ""
        
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return ""
        
        # Create model filename
        if model_name is None:
            model_name = f"model_{run_id}.pkl"
        else:
            model_name = f"{model_name}.pkl"
        
        # Copy to models directory
        stored_path = self.models_dir / model_name
        shutil.copy2(model_path, stored_path)
        
        # Update run metadata
        run = self.active_runs[run_id]
        run.model_path = str(stored_path)
        
        # Save to database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE runs SET model_path = ? WHERE run_id = ?
            """, (str(stored_path), run_id))
        
        logger.info(f"Model logged: {model_name} for run {run_id}")
        return str(stored_path)
    
    def end_run(
        self,
        run_id: str,
        status: ExperimentStatus = ExperimentStatus.COMPLETED,
        error: Optional[str] = None,
        traceback: Optional[str] = None
    ) -> None:
        """
        End an experiment run.
        
        Args:
            run_id: Run ID
            status: Final status
            error: Error message (if failed)
            traceback: Error traceback (if failed)
        """
        if run_id not in self.active_runs:
            logger.warning(f"Run {run_id} not found in active runs")
            return
        
        run = self.active_runs[run_id]
        run.status = status
        run.end_time = int(time.time() * 1000)
        run.duration_ms = run.end_time - run.start_time
        
        if error:
            run.error = error
            run.traceback = traceback
        
        # Save to database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE runs SET 
                    status = ?, end_time = ?, duration_ms = ?, error = ?, traceback = ?
                WHERE run_id = ?
            """, (
                status.value, run.end_time, run.duration_ms,
                error, traceback, run_id
            ))
        
        # Update experiment best run if completed
        if status == ExperimentStatus.COMPLETED:
            self._update_best_run(run)
        
        # Remove from active runs
        del self.active_runs[run_id]
        
        logger.info(f"Run ended: {run_id} with status {status.value}")
    
    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """
        Get experiment metadata.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            Experiment metadata or None
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM experiments WHERE experiment_id = ?
                """, (experiment_id,))
                
                row = cursor.fetchone()
                if row:
                    return self._row_to_experiment(row)
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting experiment {experiment_id}: {e}")
            return None
    
    def get_run(self, run_id: str) -> Optional[ExperimentRun]:
        """
        Get run metadata.
        
        Args:
            run_id: Run ID
            
        Returns:
            Run metadata or None
        """
        # Check active runs first
        if run_id in self.active_runs:
            return self.active_runs[run_id]
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM runs WHERE run_id = ?
                """, (run_id,))
                
                row = cursor.fetchone()
                if row:
                    return self._row_to_run(row)
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting run {run_id}: {e}")
            return None
    
    def list_experiments(
        self,
        tags: Optional[List[str]] = None,
        limit: int = 100
    ) -> List[Experiment]:
        """
        List experiments with optional filtering.
        
        Args:
            tags: Filter by tags
            limit: Maximum number of results
            
        Returns:
            List of experiments
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = "SELECT * FROM experiments WHERE 1=1"
                params = []
                
                if tags:
                    for tag in tags:
                        query += " AND tags LIKE ?"
                        params.append(f'%"{tag}"%')
                
                query += " ORDER BY created_at DESC LIMIT ?"
                params.append(limit)
                
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()
                
                return [self._row_to_experiment(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Error listing experiments: {e}")
            return []
    
    def list_runs(
        self,
        experiment_id: Optional[str] = None,
        status: Optional[ExperimentStatus] = None,
        limit: int = 100
    ) -> List[ExperimentRun]:
        """
        List runs with optional filtering.
        
        Args:
            experiment_id: Filter by experiment
            status: Filter by status
            limit: Maximum number of results
            
        Returns:
            List of runs
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = "SELECT * FROM runs WHERE 1=1"
                params = []
                
                if experiment_id:
                    query += " AND experiment_id = ?"
                    params.append(experiment_id)
                
                if status:
                    query += " AND status = ?"
                    params.append(status.value)
                
                query += " ORDER BY start_time DESC LIMIT ?"
                params.append(limit)
                
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()
                
                return [self._row_to_run(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Error listing runs: {e}")
            return []
    
    def compare_runs(
        self,
        run_ids: List[str],
        metric: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compare multiple runs.
        
        Args:
            run_ids: List of run IDs to compare
            metric: Specific metric to compare
            
        Returns:
            Comparison results
        """
        runs = []
        for run_id in run_ids:
            run = self.get_run(run_id)
            if run:
                runs.append(run)
        
        if not runs:
            return {}
        
        # Create comparison table
        comparison = {
            'runs': [],
            'metrics_comparison': {},
            'parameters_comparison': {},
            'best_run': None,
            'best_value': None
        }
        
        for run in runs:
            run_info = {
                'run_id': run.run_id,
                'name': run.name,
                'status': run.status.value,
                'duration_ms': run.duration_ms,
                'start_time': run.start_time,
                'end_time': run.end_time
            }
            
            # Add metrics
            all_metrics = {**run.metrics, **run.training_metrics, 
                          **run.validation_metrics, **run.test_metrics, **run.trading_metrics}
            
            if metric:
                run_info[metric] = all_metrics.get(metric)
            else:
                run_info['metrics'] = all_metrics
            
            # Add parameters
            run_info['parameters'] = run.parameters
            run_info['hyperparameters'] = run.hyperparameters
            
            comparison['runs'].append(run_info)
        
        # Find best run
        if metric:
            values = [(r['run_id'], r.get(metric, float('-inf'))) for r in comparison['runs']]
            best_run_id, best_value = max(values, key=lambda x: x[1])
            comparison['best_run'] = best_run_id
            comparison['best_value'] = best_value
        
        return comparison
    
    def create_comparison_plot(
        self,
        experiment_id: str,
        metric: str,
        save_path: Optional[str] = None
    ) -> str:
        """
        Create comparison plot for experiment runs.
        
        Args:
            experiment_id: Experiment ID
            metric: Metric to plot
            save_path: Path to save plot (optional)
            
        Returns:
            Path to saved plot
        """
        runs = self.list_runs(experiment_id=experiment_id, status=ExperimentStatus.COMPLETED)
        
        if not runs:
            logger.warning(f"No completed runs found for experiment {experiment_id}")
            return ""
        
        # Extract data
        run_names = []
        metric_values = []
        
        for run in runs:
            all_metrics = {**run.metrics, **run.validation_metrics, **run.trading_metrics}
            if metric in all_metrics:
                run_names.append(run.name)
                metric_values.append(all_metrics[metric])
        
        if not metric_values:
            logger.warning(f"No values found for metric {metric}")
            return ""
        
        # Create plot
        plt.figure(figsize=(12, 6))
        plt.bar(run_names, metric_values)
        plt.title(f"{metric} Comparison - Experiment {experiment_id}")
        plt.xlabel("Run")
        plt.ylabel(metric)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = self.plots_dir / f"comparison_{experiment_id}_{metric}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Comparison plot saved: {save_path}")
        return str(save_path)
    
    def _log_metric_timeseries(
        self,
        run_id: str,
        key: str,
        value: float,
        step: int,
        timestamp: Optional[int]
    ) -> None:
        """Log metric with step for time series visualization."""
        run_dir = self.runs_dir / run_id
        metrics_file = run_dir / "metrics.csv"
        
        # Create file with header if it doesn't exist
        if not metrics_file.exists():
            with open(metrics_file, 'w') as f:
                f.write("timestamp,step,key,value\n")
        
        # Append metric
        ts = timestamp or int(time.time() * 1000)
        with open(metrics_file, 'a') as f:
            f.write(f"{ts},{step},{key},{value}\n")
    
    def _get_environment_info(self) -> Dict[str, Any]:
        """Get current environment information."""
        import sys
        import platform
        
        env_info = {
            'python_version': sys.version,
            'platform': platform.platform(),
            'architecture': platform.architecture(),
            'processor': platform.processor(),
            'hostname': platform.node(),
            'timestamp': int(time.time() * 1000)
        }
        
        # Try to get git info
        try:
            import subprocess
            git_commit = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'],
                cwd=os.getcwd(),
                stderr=subprocess.DEVNULL
            ).decode('utf-8').strip()
            env_info['git_commit'] = git_commit
        except:
            pass
        
        return env_info
    
    def _update_best_run(self, run: ExperimentRun) -> None:
        """Update experiment best run if this run is better."""
        experiment = self.get_experiment(run.experiment_id)
        if not experiment:
            return
        
        # Get primary metric value
        all_metrics = {**run.metrics, **run.validation_metrics, **run.trading_metrics}
        metric_value = all_metrics.get(experiment.primary_metric)
        
        if metric_value is None:
            return
        
        # Check if this is the best run
        current_best = experiment.best_value
        if current_best is None:
            is_better = True
        elif experiment.metric_direction == "maximize":
            is_better = metric_value > current_best
        else:
            is_better = metric_value < current_best
        
        if is_better:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE experiments SET best_run_id = ?, best_value = ?
                    WHERE experiment_id = ?
                """, (run.run_id, metric_value, run.experiment_id))
            
            logger.info(f"New best run for experiment {run.experiment_id}: {run.run_id}")
    
    def _row_to_experiment(self, row: tuple) -> Experiment:
        """Convert database row to Experiment."""
        return Experiment(
            experiment_id=row[0],
            name=row[1],
            description=row[2],
            created_at=row[3],
            created_by=row[4],
            tags=json.loads(row[5]) if row[5] else [],
            best_run_id=row[6],
            best_metric=row[7],
            best_value=row[8],
            metric_direction=row[9],
            primary_metric=row[10]
        )
    
    def _row_to_run(self, row: tuple) -> ExperimentRun:
        """Convert database row to ExperimentRun."""
        return ExperimentRun(
            run_id=row[0],
            experiment_id=row[1],
            name=row[2],
            status=ExperimentStatus(row[3]),
            start_time=row[4],
            end_time=row[5],
            duration_ms=row[6],
            parameters=json.loads(row[7]) if row[7] else {},
            hyperparameters=json.loads(row[8]) if row[8] else {},
            metrics=json.loads(row[9]) if row[9] else {},
            training_metrics=json.loads(row[10]) if row[10] else {},
            validation_metrics=json.loads(row[11]) if row[11] else {},
            test_metrics=json.loads(row[12]) if row[12] else {},
            trading_metrics=json.loads(row[13]) if row[13] else {},
            model_path=row[14],
            artifacts=json.loads(row[15]) if row[15] else {},
            git_commit=row[16],
            environment=json.loads(row[17]) if row[17] else {},
            error=row[18],
            traceback=row[19]
        )


# Convenience functions
def create_experiment_tracker(base_path: str = "mlflow_registry") -> ExperimentTracker:
    """Create experiment tracker with default settings."""
    return ExperimentTracker(base_path)


def create_experiment_context(
    tracker: ExperimentTracker,
    experiment_id: str,
    run_name: str,
    parameters: Optional[Dict[str, Any]] = None
):
    """Context manager for experiment runs."""
    class ExperimentContext:
        def __init__(self, tracker, experiment_id, run_name, parameters):
            self.tracker = tracker
            self.experiment_id = experiment_id
            self.run_name = run_name
            self.parameters = parameters or {}
            self.run_id = None
        
        def __enter__(self):
            self.run_id = self.tracker.start_run(
                experiment_id=self.experiment_id,
                run_name=self.run_name,
                parameters=self.parameters
            )
            return self.run_id
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_type is not None:
                self.tracker.end_run(
                    run_id=self.run_id,
                    status=ExperimentStatus.FAILED,
                    error=str(exc_val),
                    traceback=str(exc_tb)
                )
            else:
                self.tracker.end_run(run_id=self.run_id)
    
    return ExperimentContext(tracker, experiment_id, run_name, parameters)


if __name__ == "__main__":
    # Test experiment tracker
    logging.basicConfig(level=logging.INFO)
    
    tracker = create_experiment_tracker("test_tracking")
    
    # Create experiment
    exp_id = tracker.create_experiment(
        name="LSTM Trading Strategy",
        description="Test LSTM strategy for BTC/USDT",
        tags=["lstm", "trading", "btc"],
        primary_metric="sharpe_ratio"
    )
    
    print(f"Created experiment: {exp_id}")
    
    # Start run with context manager
    with create_experiment_context(
        tracker=tracker,
        experiment_id=exp_id,
        run_name="test_run_1",
        parameters={"learning_rate": 0.001, "hidden_size": 64}
    ) as run_id:
        
        print(f"Started run: {run_id}")
        
        # Log some metrics
        tracker.log_metric(run_id, "accuracy", 0.85)
        tracker.log_metric(run_id, "sharpe_ratio", 1.25)
        tracker.log_metrics(run_id, {"precision": 0.82, "recall": 0.88}, "validation")
        tracker.log_metrics(run_id, {"win_rate": 0.65, "profit_factor": 2.1}, "trading")
        
        # Log parameter
        tracker.log_parameter(run_id, "model_type", "LSTM")
        
        print("Metrics and parameters logged")
    
    # Get experiment info
    experiment = tracker.get_experiment(exp_id)
    if experiment:
        print(f"Experiment: {experiment.name}")
        print(f"Best run: {experiment.best_run_id}")
        print(f"Best value: {experiment.best_value}")
    
    # List runs
    runs = tracker.list_runs(experiment_id=exp_id)
    print(f"Total runs: {len(runs)}")
    
    for run in runs[:3]:  # Show first 3
        print(f"  Run: {run.name} - Status: {run.status.value}")
        if run.metrics:
            print(f"    Sharpe: {run.metrics.get('sharpe_ratio', 'N/A')}")
