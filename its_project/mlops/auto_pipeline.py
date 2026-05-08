#!/usr/bin/env python3
"""
MLOps Automated Pipeline
========================

Production-ready automated ML pipeline:
- Automated training
- Model evaluation
- Performance validation
- Automated deployment
- Rollback capabilities
- Pipeline orchestration
"""

from __future__ import annotations

import os
import json
import logging
import time
import asyncio
import subprocess
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import sqlite3
from enum import Enum
import yaml
import pickle
import numpy as np

from .versioning import VersionManager, ModelMetadata
from .experiment_tracking import ExperimentTracker, ExperimentStatus
from .drift_detection import DriftDetector, DriftType

logger = logging.getLogger(__name__)


class PipelineStatus(Enum):
    """Pipeline execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    DEPLOYED = "deployed"


class DeploymentStage(Enum):
    """Deployment stages."""
    TRAINING = "training"
    EVALUATION = "evaluation"
    VALIDATION = "validation"
    DEPLOYMENT = "deployment"
    MONITORING = "monitoring"


@dataclass
class PipelineConfig:
    """Pipeline configuration."""
    pipeline_id: str
    name: str
    description: str
    
    # Training configuration
    training_script: str
    training_params: Dict[str, Any] = field(default_factory=dict)
    data_config: Dict[str, Any] = field(default_factory=dict)
    
    # Evaluation configuration
    evaluation_script: str
    evaluation_metrics: List[str] = field(default_factory=list)
    validation_thresholds: Dict[str, float] = field(default_factory=dict)
    
    # Deployment configuration
    deployment_env: str = "production"
    deployment_script: Optional[str] = None
    rollback_enabled: bool = True
    
    # Scheduling
    schedule_enabled: bool = False
    schedule_cron: Optional[str] = None
    auto_retrain_on_drift: bool = True
    
    # Resources
    cpu_limit: Optional[int] = None
    memory_limit: Optional[str] = None
    gpu_enabled: bool = False


@dataclass
class PipelineRun:
    """Pipeline execution run."""
    run_id: str
    pipeline_id: str
    status: PipelineStatus
    start_time: int
    end_time: Optional[int] = None
    duration_ms: Optional[int] = None
    
    # Stage tracking
    current_stage: Optional[DeploymentStage] = None
    completed_stages: List[DeploymentStage] = field(default_factory=list)
    stage_results: Dict[str, Any] = field(default_factory=dict)
    
    # Results
    model_id: Optional[str] = None
    model_version: Optional[str] = None
    evaluation_results: Dict[str, float] = field(default_factory=dict)
    deployment_results: Dict[str, Any] = field(default_factory=dict)
    
    # Error handling
    error: Optional[str] = None
    traceback: Optional[str] = None
    
    # Metadata
    triggered_by: str = "manual"
    git_commit: Optional[str] = None


class AutoPipeline:
    """
    Production-ready automated ML pipeline.
    
    Features:
    - Automated training and evaluation
    - Model validation and deployment
    - Rollback capabilities
    - Scheduling and triggers
    - Integration with MLOps components
    """
    
    def __init__(
        self,
        base_path: str = "ml_pipeline",
        version_manager: Optional[VersionManager] = None,
        experiment_tracker: Optional[ExperimentTracker] = None,
        drift_detector: Optional[DriftDetector] = None
    ) -> None:
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Directory structure
        self.pipelines_dir = self.base_path / "pipelines"
        self.runs_dir = self.base_path / "runs"
        self.models_dir = self.base_path / "models"
        self.logs_dir = self.base_path / "logs"
        self.db_path = self.base_path / "pipeline.db"
        
        # Create directories
        for directory in [self.pipelines_dir, self.runs_dir, self.models_dir, self.logs_dir]:
            directory.mkdir(exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # MLOps components
        self.version_manager = version_manager or VersionManager(str(self.base_path / "registry"))
        self.experiment_tracker = experiment_tracker or ExperimentTracker(str(self.base_path / "experiments"))
        self.drift_detector = drift_detector or DriftDetector(str(self.base_path / "drift"))
        
        # Active runs
        self.active_runs: Dict[str, PipelineRun] = {}
        
        # Pipeline configurations
        self.pipeline_configs: Dict[str, PipelineConfig] = {}
        
        # Callbacks
        self.pipeline_callbacks: Dict[str, List[Callable]] = {
            'pipeline_started': [],
            'pipeline_completed': [],
            'pipeline_failed': [],
            'stage_completed': [],
            'model_deployed': [],
            'rollback_triggered': []
        }
        
        # Statistics
        self.stats = {
            'total_runs': 0,
            'successful_runs': 0,
            'failed_runs': 0,
            'models_deployed': 0,
            'rollbacks_triggered': 0,
            'auto_retrains': 0
        }
        
        # Background tasks
        self.scheduler_task: Optional[asyncio.Task] = None
        self.monitoring_task: Optional[asyncio.Task] = None
        self.running = False
    
    def _init_database(self) -> None:
        """Initialize SQLite database for pipeline tracking."""
        with sqlite3.connect(self.db_path) as conn:
            # Pipelines table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS pipelines (
                    pipeline_id TEXT PRIMARY KEY,
                    name TEXT,
                    description TEXT,
                    config TEXT,
                    created_at INTEGER,
                    updated_at INTEGER,
                    active BOOLEAN
                )
            """)
            
            # Pipeline runs table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS pipeline_runs (
                    run_id TEXT PRIMARY KEY,
                    pipeline_id TEXT,
                    status TEXT,
                    start_time INTEGER,
                    end_time INTEGER,
                    duration_ms INTEGER,
                    current_stage TEXT,
                    completed_stages TEXT,
                    stage_results TEXT,
                    model_id TEXT,
                    model_version TEXT,
                    evaluation_results TEXT,
                    deployment_results TEXT,
                    error TEXT,
                    traceback TEXT,
                    triggered_by TEXT,
                    git_commit TEXT,
                    FOREIGN KEY (pipeline_id) REFERENCES pipelines (pipeline_id)
                )
            """)
            
            # Indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_pipeline ON pipeline_runs(pipeline_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_status ON pipeline_runs(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_start_time ON pipeline_runs(start_time)")
    
    def create_pipeline(self, config: PipelineConfig) -> str:
        """
        Create a new pipeline configuration.
        
        Args:
            config: Pipeline configuration
            
        Returns:
            Pipeline ID
        """
        # Save configuration
        config_path = self.pipelines_dir / f"{config.pipeline_id}.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(asdict(config), f, default_flow_style=False)
        
        # Save to database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO pipelines VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                config.pipeline_id, config.name, config.description,
                json.dumps(asdict(config)), int(time.time() * 1000),
                int(time.time() * 1000), True
            ))
        
        self.pipeline_configs[config.pipeline_id] = config
        
        logger.info(f"Pipeline created: {config.name} ({config.pipeline_id})")
        return config.pipeline_id
    
    def run_pipeline(
        self,
        pipeline_id: str,
        triggered_by: str = "manual",
        force: bool = False
    ) -> str:
        """
        Execute a pipeline run.
        
        Args:
            pipeline_id: Pipeline ID
            triggered_by: Trigger source
            force: Force run even if already running
            
        Returns:
            Run ID
        """
        # Get pipeline configuration
        config = self.get_pipeline(pipeline_id)
        if not config:
            raise ValueError(f"Pipeline {pipeline_id} not found")
        
        # Check if already running
        if not force:
            active_runs = [r for r in self.active_runs.values() if r.pipeline_id == pipeline_id]
            if active_runs:
                raise ValueError(f"Pipeline {pipeline_id} already running")
        
        # Create run
        run_id = f"{pipeline_id}_{int(time.time())}"
        run = PipelineRun(
            run_id=run_id,
            pipeline_id=pipeline_id,
            status=PipelineStatus.RUNNING,
            start_time=int(time.time() * 1000),
            triggered_by=triggered_by,
            git_commit=self._get_git_commit()
        )
        
        # Save to database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO pipeline_runs VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            """, (
                run.run_id, run.pipeline_id, run.status.value,
                run.start_time, run.end_time, run.duration_ms,
                run.current_stage.value if run.current_stage else None,
                json.dumps([s.value for s in run.completed_stages]),
                json.dumps(run.stage_results), run.model_id, run.model_version,
                json.dumps(run.evaluation_results), json.dumps(run.deployment_results),
                run.error, run.traceback, run.triggered_by, run.git_commit
            ))
        
        # Add to active runs
        self.active_runs[run_id] = run
        
        # Update statistics
        self.stats['total_runs'] += 1
        
        # Trigger callbacks
        self._trigger_callbacks('pipeline_started', run)
        
        # Start pipeline execution
        asyncio.create_task(self._execute_pipeline(run_id))
        
        logger.info(f"Pipeline started: {pipeline_id} ({run_id})")
        return run_id
    
    async def _execute_pipeline(self, run_id: str) -> None:
        """Execute pipeline stages."""
        run = self.active_runs.get(run_id)
        if not run:
            return
        
        config = self.get_pipeline(run.pipeline_id)
        if not config:
            return
        
        try:
            # Stage 1: Training
            await self._execute_training_stage(run, config)
            
            # Stage 2: Evaluation
            await self._execute_evaluation_stage(run, config)
            
            # Stage 3: Validation
            await self._execute_validation_stage(run, config)
            
            # Stage 4: Deployment
            await self._execute_deployment_stage(run, config)
            
            # Stage 5: Monitoring
            await self._execute_monitoring_stage(run, config)
            
            # Mark as completed
            run.status = PipelineStatus.COMPLETED
            run.end_time = int(time.time() * 1000)
            run.duration_ms = run.end_time - run.start_time
            
            # Update statistics
            self.stats['successful_runs'] += 1
            
            logger.info(f"Pipeline completed: {run.run_id}")
            
        except Exception as e:
            # Handle pipeline failure
            run.status = PipelineStatus.FAILED
            run.error = str(e)
            run.traceback = str(e.__traceback__)
            run.end_time = int(time.time() * 1000)
            run.duration_ms = run.end_time - run.start_time
            
            # Update statistics
            self.stats['failed_runs'] += 1
            
            logger.error(f"Pipeline failed: {run.run_id} - {e}")
        
        finally:
            # Update database
            await self._update_run_in_database(run)
            
            # Remove from active runs
            if run_id in self.active_runs:
                del self.active_runs[run_id]
            
            # Trigger callbacks
            if run.status == PipelineStatus.COMPLETED:
                self._trigger_callbacks('pipeline_completed', run)
            else:
                self._trigger_callbacks('pipeline_failed', run)
    
    async def _execute_training_stage(self, run: PipelineRun, config: PipelineConfig) -> None:
        """Execute training stage."""
        run.current_stage = DeploymentStage.TRAINING
        await self._update_run_in_database(run)
        
        logger.info(f"Starting training stage for {run.run_id}")
        
        # Create experiment
        experiment_id = self.experiment_tracker.create_experiment(
            name=f"Auto Pipeline - {config.name}",
            description=f"Automated training for pipeline {config.pipeline_id}",
            tags=["auto_pipeline", "training"],
            primary_metric="sharpe_ratio"
        )
        
        # Start experiment run
        with self.experiment_tracker.create_experiment_context(
            experiment_id=experiment_id,
            run_name=f"auto_train_{run.run_id}",
            parameters=config.training_params
        ) as exp_run_id:
            
            # Prepare training command
            cmd = [
                "python", config.training_script,
                "--config", json.dumps(config.data_config),
                "--params", json.dumps(config.training_params),
                "--output", str(self.models_dir / f"{run.run_id}_model.pkl"),
                "--experiment_id", experiment_id,
                "--run_id", exp_run_id
            ]
            
            # Execute training
            result = await self._run_command(cmd, f"training_{run.run_id}")
            
            if result['returncode'] != 0:
                raise RuntimeError(f"Training failed: {result['stderr']}")
            
            # Parse training results
            try:
                with open(self.models_dir / f"{run.run_id}_results.json", 'r') as f:
                    training_results = json.load(f)
            except:
                training_results = {}
            
            # Log metrics
            for metric, value in training_results.items():
                if isinstance(value, (int, float)):
                    self.experiment_tracker.log_metric(exp_run_id, metric, value)
            
            # Save model
            model_path = self.models_dir / f"{run.run_id}_model.pkl"
            if model_path.exists():
                logged_path = self.experiment_tracker.log_model(exp_run_id, str(model_path))
                run.stage_results['training'] = {
                    'experiment_id': experiment_id,
                    'run_id': exp_run_id,
                    'model_path': logged_path,
                    'results': training_results
                }
        
        run.completed_stages.append(DeploymentStage.TRAINING)
        await self._update_run_in_database(run)
        self._trigger_callbacks('stage_completed', run, 'training')
        
        logger.info(f"Training stage completed for {run.run_id}")
    
    async def _execute_evaluation_stage(self, run: PipelineRun, config: PipelineConfig) -> None:
        """Execute evaluation stage."""
        run.current_stage = DeploymentStage.EVALUATION
        await self._update_run_in_database(run)
        
        logger.info(f"Starting evaluation stage for {run.run_id}")
        
        # Get model from training stage
        model_path = run.stage_results.get('training', {}).get('model_path')
        if not model_path:
            raise RuntimeError("No model found from training stage")
        
        # Prepare evaluation command
        cmd = [
            "python", config.evaluation_script,
            "--model", model_path,
            "--config", json.dumps(config.data_config),
            "--metrics", json.dumps(config.evaluation_metrics),
            "--output", str(self.runs_dir / f"{run.run_id}_evaluation.json")
        ]
        
        # Execute evaluation
        result = await self._run_command(cmd, f"evaluation_{run.run_id}")
        
        if result['returncode'] != 0:
            raise RuntimeError(f"Evaluation failed: {result['stderr']}")
        
        # Parse evaluation results
        try:
            with open(self.runs_dir / f"{run.run_id}_evaluation.json", 'r') as f:
                evaluation_results = json.load(f)
        except:
            evaluation_results = {}
        
        run.evaluation_results = evaluation_results
        run.stage_results['evaluation'] = evaluation_results
        
        run.completed_stages.append(DeploymentStage.EVALUATION)
        await self._update_run_in_database(run)
        self._trigger_callbacks('stage_completed', run, 'evaluation')
        
        logger.info(f"Evaluation stage completed for {run.run_id}")
    
    async def _execute_validation_stage(self, run: PipelineRun, config: PipelineConfig) -> None:
        """Execute validation stage."""
        run.current_stage = DeploymentStage.VALIDATION
        await self._update_run_in_database(run)
        
        logger.info(f"Starting validation stage for {run.run_id}")
        
        # Check validation thresholds
        validation_passed = True
        validation_results = {}
        
        for metric, threshold in config.validation_thresholds.items():
            if metric in run.evaluation_results:
                value = run.evaluation_results[metric]
                validation_results[metric] = {
                    'value': value,
                    'threshold': threshold,
                    'passed': value >= threshold
                }
                
                if value < threshold:
                    validation_passed = False
        
        if not validation_passed:
            raise RuntimeError(f"Validation failed: {validation_results}")
        
        run.stage_results['validation'] = {
            'passed': validation_passed,
            'results': validation_results
        }
        
        run.completed_stages.append(DeploymentStage.VALIDATION)
        await self._update_run_in_database(run)
        self._trigger_callbacks('stage_completed', run, 'validation')
        
        logger.info(f"Validation stage completed for {run.run_id}")
    
    async def _execute_deployment_stage(self, run: PipelineRun, config: PipelineConfig) -> None:
        """Execute deployment stage."""
        run.current_stage = DeploymentStage.DEPLOYMENT
        await self._update_run_in_database(run)
        
        logger.info(f"Starting deployment stage for {run.run_id}")
        
        # Get model path
        model_path = run.stage_results.get('training', {}).get('model_path')
        if not model_path:
            raise RuntimeError("No model found for deployment")
        
        # Create model metadata
        model_metadata = self.version_manager.create_model_metadata(
            model_id=f"{config.pipeline_id}_model",
            version=self.version_manager.create_version("1.0.0", "minor"),
            model_type="auto_pipeline",
            framework="pytorch",
            description=f"Auto-generated model from pipeline {config.pipeline_id}",
            created_by="auto_pipeline"
        )
        
        # Add performance metrics
        model_metadata.sharpe_ratio = run.evaluation_results.get('sharpe_ratio')
        model_metadata.max_drawdown = run.evaluation_results.get('max_drawdown')
        model_metadata.win_rate = run.evaluation_results.get('win_rate')
        model_metadata.training_config = config.training_params
        model_metadata.validation_metrics = run.evaluation_results
        
        # Register model
        if self.version_manager.register_model(f"{config.pipeline_id}_model", model_path, model_metadata):
            run.model_id = model_metadata.model_id
            run.model_version = model_metadata.version
            
            # Deploy model
            if self.version_manager.deploy_model(
                model_id=model_metadata.model_id,
                version=model_metadata.version,
                deployment_env=config.deployment_env
            ):
                run.stage_results['deployment'] = {
                    'model_id': model_metadata.model_id,
                    'version': model_metadata.version,
                    'environment': config.deployment_env,
                    'deployed': True
                }
                
                # Update statistics
                self.stats['models_deployed'] += 1
                
                # Trigger callbacks
                self._trigger_callbacks('model_deployed', run, model_metadata)
                
                logger.info(f"Model deployed: {model_metadata.model_id} v{model_metadata.version}")
            else:
                raise RuntimeError("Model deployment failed")
        else:
            raise RuntimeError("Model registration failed")
        
        run.completed_stages.append(DeploymentStage.DEPLOYMENT)
        await self._update_run_in_database(run)
        self._trigger_callbacks('stage_completed', run, 'deployment')
        
        logger.info(f"Deployment stage completed for {run.run_id}")
    
    async def _execute_monitoring_stage(self, run: PipelineRun, config: PipelineConfig) -> None:
        """Execute monitoring stage."""
        run.current_stage = DeploymentStage.MONITORING
        await self._update_run_in_database(run)
        
        logger.info(f"Starting monitoring stage for {run.run_id}")
        
        # Set up drift detection for deployed model
        if run.model_id and config.auto_retrain_on_drift:
            # This would integrate with drift detector
            # For now, just log the setup
            run.stage_results['monitoring'] = {
                'drift_detection_enabled': True,
                'model_id': run.model_id,
                'auto_retrain_enabled': config.auto_retrain_on_drift
            }
        
        run.completed_stages.append(DeploymentStage.MONITORING)
        await self._update_run_in_database(run)
        self._trigger_callbacks('stage_completed', run, 'monitoring')
        
        logger.info(f"Monitoring stage completed for {run.run_id}")
    
    async def _run_command(self, cmd: List[str], log_prefix: str) -> Dict[str, Any]:
        """Run command and capture output."""
        log_file = self.logs_dir / f"{log_prefix}.log"
        
        with open(log_file, 'w') as log:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.base_path)
            )
            
            stdout, stderr = await process.communicate()
            
            # Write to log file
            log.write(f"Command: {' '.join(cmd)}\n")
            log.write(f"Return code: {process.returncode}\n")
            log.write(f"STDOUT:\n{stdout.decode()}\n")
            log.write(f"STDERR:\n{stderr.decode()}\n")
        
        return {
            'returncode': process.returncode,
            'stdout': stdout.decode(),
            'stderr': stderr.decode(),
            'log_file': str(log_file)
        }
    
    async def _update_run_in_database(self, run: PipelineRun) -> None:
        """Update run in database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE pipeline_runs SET
                    status = ?, current_stage = ?, completed_stages = ?,
                    stage_results = ?, model_id = ?, model_version = ?,
                    evaluation_results = ?, deployment_results = ?,
                    error = ?, traceback = ?, end_time = ?, duration_ms = ?
                WHERE run_id = ?
            """, (
                run.status.value,
                run.current_stage.value if run.current_stage else None,
                json.dumps([s.value for s in run.completed_stages]),
                json.dumps(run.stage_results), run.model_id, run.model_version,
                json.dumps(run.evaluation_results), json.dumps(run.deployment_results),
                run.error, run.traceback, run.end_time, run.duration_ms,
                run.run_id
            ))
    
    def rollback_model(self, pipeline_id: str, target_version: Optional[str] = None) -> bool:
        """
        Rollback model to previous version.
        
        Args:
            pipeline_id: Pipeline ID
            target_version: Target version (None for previous)
            
        Returns:
            True if rollback successful
        """
        model_id = f"{pipeline_id}_model"
        
        if self.version_manager.rollback_model(model_id, target_version):
            self.stats['rollbacks_triggered'] += 1
            
            # Trigger callbacks
            self._trigger_callbacks('rollback_triggered', {
                'pipeline_id': pipeline_id,
                'model_id': model_id,
                'target_version': target_version
            })
            
            logger.info(f"Model rolled back: {model_id}")
            return True
        
        return False
    
    def get_pipeline(self, pipeline_id: str) -> Optional[PipelineConfig]:
        """Get pipeline configuration."""
        # Check cache first
        if pipeline_id in self.pipeline_configs:
            return self.pipeline_configs[pipeline_id]
        
        # Load from file
        config_path = self.pipelines_dir / f"{pipeline_id}.yaml"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
                config = PipelineConfig(**config_dict)
                self.pipeline_configs[pipeline_id] = config
                return config
        
        return None
    
    def get_run(self, run_id: str) -> Optional[PipelineRun]:
        """Get pipeline run."""
        # Check active runs first
        if run_id in self.active_runs:
            return self.active_runs[run_id]
        
        # Load from database
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM pipeline_runs WHERE run_id = ?
                """, (run_id,))
                
                row = cursor.fetchone()
                if row:
                    return self._row_to_run(row)
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting run {run_id}: {e}")
            return None
    
    def list_pipelines(self, active_only: bool = False) -> List[PipelineConfig]:
        """List pipeline configurations."""
        pipelines = []
        
        for config_file in self.pipelines_dir.glob("*.yaml"):
            try:
                with open(config_file, 'r') as f:
                    config_dict = yaml.safe_load(f)
                    config = PipelineConfig(**config_dict)
                    
                    if not active_only or self._is_pipeline_active(config.pipeline_id):
                        pipelines.append(config)
            except Exception as e:
                logger.error(f"Error loading pipeline {config_file}: {e}")
        
        return pipelines
    
    def list_runs(
        self,
        pipeline_id: Optional[str] = None,
        status: Optional[PipelineStatus] = None,
        limit: int = 100
    ) -> List[PipelineRun]:
        """List pipeline runs."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = "SELECT * FROM pipeline_runs WHERE 1=1"
                params = []
                
                if pipeline_id:
                    query += " AND pipeline_id = ?"
                    params.append(pipeline_id)
                
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
    
    def start_scheduler(self) -> None:
        """Start pipeline scheduler."""
        if self.scheduler_task and not self.scheduler_task.done():
            logger.warning("Scheduler already running")
            return
        
        self.running = True
        self.scheduler_task = asyncio.create_task(self._scheduler_loop())
        logger.info("Pipeline scheduler started")
    
    def stop_scheduler(self) -> None:
        """Stop pipeline scheduler."""
        self.running = False
        
        if self.scheduler_task:
            self.scheduler_task.cancel()
            try:
                asyncio.run(self.scheduler_task)
            except asyncio.CancelledError:
                pass
        
        logger.info("Pipeline scheduler stopped")
    
    async def _scheduler_loop(self) -> None:
        """Background scheduler loop."""
        logger.info("Scheduler loop started")
        
        while self.running:
            try:
                # Check scheduled pipelines
                pipelines = self.list_pipelines(active_only=True)
                
                for pipeline in pipelines:
                    if pipeline.schedule_enabled and self._should_run_pipeline(pipeline):
                        await self._run_scheduled_pipeline(pipeline)
                
                # Sleep for next check
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                await asyncio.sleep(5)
        
        logger.info("Scheduler loop stopped")
    
    def _should_run_pipeline(self, config: PipelineConfig) -> bool:
        """Check if pipeline should run based on schedule."""
        # Simplified scheduling logic
        # In production, use cron parser
        if not config.schedule_cron:
            return False
        
        # For now, just check if it's a new hour
        current_time = datetime.now()
        return current_time.minute == 0
    
    async def _run_scheduled_pipeline(self, config: PipelineConfig) -> None:
        """Run scheduled pipeline."""
        try:
            self.run_pipeline(
                pipeline_id=config.pipeline_id,
                triggered_by="scheduler"
            )
            
            self.stats['auto_retrains'] += 1
            
        except Exception as e:
            logger.error(f"Scheduled pipeline failed {config.pipeline_id}: {e}")
    
    def _is_pipeline_active(self, pipeline_id: str) -> bool:
        """Check if pipeline is active."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT active FROM pipelines WHERE pipeline_id = ?
                """, (pipeline_id,))
                
                row = cursor.fetchone()
                return bool(row[0]) if row else False
                
        except Exception:
            return False
    
    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit."""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                cwd=os.getcwd()
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except:
            return None
    
    def _trigger_callbacks(self, event: str, *args) -> None:
        """Trigger pipeline event callbacks."""
        for callback in self.pipeline_callbacks.get(event, []):
            try:
                callback(*args)
            except Exception as e:
                logger.error(f"Error in {event} callback: {e}")
    
    def add_callback(self, event: str, callback: Callable) -> None:
        """Add callback for pipeline events."""
        if event not in self.pipeline_callbacks:
            self.pipeline_callbacks[event] = []
        self.pipeline_callbacks[event].append(callback)
    
    def _row_to_run(self, row: tuple) -> PipelineRun:
        """Convert database row to PipelineRun."""
        return PipelineRun(
            run_id=row[0],
            pipeline_id=row[1],
            status=PipelineStatus(row[2]),
            start_time=row[3],
            end_time=row[4],
            duration_ms=row[5],
            current_stage=DeploymentStage(row[6]) if row[6] else None,
            completed_stages=[DeploymentStage(s) for s in json.loads(row[7]) if row[7]],
            stage_results=json.loads(row[8]) if row[8] else {},
            model_id=row[9],
            model_version=row[10],
            evaluation_results=json.loads(row[11]) if row[11] else {},
            deployment_results=json.loads(row[12]) if row[12] else {},
            error=row[13],
            traceback=row[14],
            triggered_by=row[15],
            git_commit=row[16]
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        base_stats = self.stats.copy()
        
        # Add current status
        base_stats.update({
            'active_pipelines': len([p for p in self.list_pipelines() if self._is_pipeline_active(p.pipeline_id)]),
            'active_runs': len(self.active_runs),
            'total_pipelines': len(self.list_pipelines()),
            'scheduler_running': self.running
        })
        
        return base_stats


# Convenience functions
def create_auto_pipeline(
    base_path: str = "ml_pipeline",
    version_manager: Optional[VersionManager] = None,
    experiment_tracker: Optional[ExperimentTracker] = None,
    drift_detector: Optional[DriftDetector] = None
) -> AutoPipeline:
    """Create auto pipeline with default settings."""
    return AutoPipeline(
        base_path=base_path,
        version_manager=version_manager,
        experiment_tracker=experiment_tracker,
        drift_detector=drift_detector
    )


def create_pipeline_config(
    pipeline_id: str,
    name: str,
    training_script: str,
    evaluation_script: str,
    description: str = ""
) -> PipelineConfig:
    """Create pipeline configuration with common settings."""
    return PipelineConfig(
        pipeline_id=pipeline_id,
        name=name,
        description=description,
        training_script=training_script,
        evaluation_script=evaluation_script,
        evaluation_metrics=["sharpe_ratio", "max_drawdown", "win_rate", "profit_factor"],
        validation_thresholds={
            "sharpe_ratio": 1.0,
            "win_rate": 0.55,
            "profit_factor": 1.5
        },
        auto_retrain_on_drift=True
    )


if __name__ == "__main__":
    # Test auto pipeline
    logging.basicConfig(level=logging.INFO)
    
    pipeline = create_auto_pipeline("test_pipeline")
    
    # Create test pipeline configuration
    config = create_pipeline_config(
        pipeline_id="test_lstm_pipeline",
        name="LSTM Trading Pipeline",
        training_script="train_model.py",
        evaluation_script="evaluate_model.py",
        description="Automated LSTM trading model pipeline"
    )
    
    # Create pipeline
    pipeline_id = pipeline.create_pipeline(config)
    print(f"Created pipeline: {pipeline_id}")
    
    # List pipelines
    pipelines = pipeline.list_pipelines()
    print(f"Total pipelines: {len(pipelines)}")
    
    # Get statistics
    stats = pipeline.get_statistics()
    print(f"Pipeline statistics: {stats}")
    
    print("Auto pipeline system ready!")
