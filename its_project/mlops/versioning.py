#!/usr/bin/env python3
"""
MLOps Versioning System
======================

Production-ready versioning for ML models and data:
- Model versioning with metadata
- Data versioning and lineage
- Experiment tracking
- Model registry
- Rollback capabilities
"""

from __future__ import annotations

import os
import json
import hashlib
import logging
import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
import pickle
import shutil
import sqlite3
from enum import Enum

logger = logging.getLogger(__name__)


class VersionType(Enum):
    """Version types."""
    MAJOR = "major"
    MINOR = "minor"
    PATCH = "patch"
    EXPERIMENT = "experiment"


@dataclass
class ModelMetadata:
    """Model version metadata."""
    model_id: str
    version: str
    version_type: VersionType
    created_at: int
    created_by: str
    description: str
    model_type: str  # "lstm", "transformer", "ensemble"
    framework: str   # "pytorch", "tensorflow", "sklearn"
    file_path: str
    file_hash: str
    file_size: int
    
    # Performance metrics
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    
    # Trading-specific metrics
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    win_rate: Optional[float] = None
    profit_factor: Optional[float] = None
    
    # Training metadata
    training_data_version: Optional[str] = None
    training_config: Dict[str, Any] = field(default_factory=dict)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    
    # Deployment metadata
    deployed: bool = False
    deployment_env: Optional[str] = None
    deployment_date: Optional[int] = None
    
    # Validation metadata
    validation_metrics: Dict[str, Any] = field(default_factory=dict)
    backtest_results: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataMetadata:
    """Data version metadata."""
    data_id: str
    version: str
    version_type: VersionType
    created_at: int
    created_by: str
    description: str
    
    # Data characteristics
    data_type: str  # "market_data", "features", "labels"
    source: str    # "binance", "polygon", etc.
    symbols: List[str] = field(default_factory=list)
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    
    # File information
    file_path: str
    file_hash: str
    file_size: int
    file_format: str  # "parquet", "csv", "hdf5"
    
    # Data quality metrics
    total_records: int = 0
    missing_values: int = 0
    duplicate_records: int = 0
    quality_score: Optional[float] = None
    
    # Processing metadata
    preprocessing_steps: List[str] = field(default_factory=list)
    feature_columns: List[str] = field(default_factory=list)
    target_columns: List[str] = field(default_factory=list)


class VersionManager:
    """
    Production-ready version management system.
    
    Features:
    - Model and data versioning
    - Metadata tracking
    - File integrity verification
    - Rollback capabilities
    - Search and filtering
    """
    
    def __init__(self, base_path: str = "mlops_registry") -> None:
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Directory structure
        self.models_dir = self.base_path / "models"
        self.data_dir = self.base_path / "data"
        self.db_path = self.base_path / "registry.db"
        
        # Create directories
        self.models_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # Cache
        self._model_cache: Dict[str, ModelMetadata] = {}
        self._data_cache: Dict[str, DataMetadata] = {}
    
    def _init_database(self) -> None:
        """Initialize SQLite database for metadata."""
        with sqlite3.connect(self.db_path) as conn:
            # Models table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS models (
                    model_id TEXT PRIMARY KEY,
                    version TEXT,
                    version_type TEXT,
                    created_at INTEGER,
                    created_by TEXT,
                    description TEXT,
                    model_type TEXT,
                    framework TEXT,
                    file_path TEXT,
                    file_hash TEXT,
                    file_size INTEGER,
                    accuracy REAL,
                    precision REAL,
                    recall REAL,
                    f1_score REAL,
                    sharpe_ratio REAL,
                    max_drawdown REAL,
                    win_rate REAL,
                    profit_factor REAL,
                    training_data_version TEXT,
                    training_config TEXT,
                    hyperparameters TEXT,
                    deployed BOOLEAN,
                    deployment_env TEXT,
                    deployment_date INTEGER,
                    validation_metrics TEXT,
                    backtest_results TEXT
                )
            """)
            
            # Data table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS data_versions (
                    data_id TEXT PRIMARY KEY,
                    version TEXT,
                    version_type TEXT,
                    created_at INTEGER,
                    created_by TEXT,
                    description TEXT,
                    data_type TEXT,
                    source TEXT,
                    symbols TEXT,
                    start_date TEXT,
                    end_date TEXT,
                    file_path TEXT,
                    file_hash TEXT,
                    file_size INTEGER,
                    file_format TEXT,
                    total_records INTEGER,
                    missing_values INTEGER,
                    duplicate_records INTEGER,
                    quality_score REAL,
                    preprocessing_steps TEXT,
                    feature_columns TEXT,
                    target_columns TEXT
                )
            """)
            
            # Indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_models_version ON models(version)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_models_type ON models(model_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_data_version ON data_versions(version)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_data_type ON data_versions(data_type)")
    
    def register_model(
        self,
        model_id: str,
        model_file: str,
        metadata: ModelMetadata
    ) -> bool:
        """
        Register a new model version.
        
        Args:
            model_id: Unique model identifier
            model_file: Path to model file
            metadata: Model metadata
            
        Returns:
            True if registration successful
        """
        try:
            # Validate model file
            if not os.path.exists(model_file):
                logger.error(f"Model file not found: {model_file}")
                return False
            
            # Calculate file hash and size
            file_hash = self._calculate_file_hash(model_file)
            file_size = os.path.getsize(model_file)
            
            # Copy model to registry
            model_filename = f"{model_id}_{metadata.version}.pkl"
            registry_path = self.models_dir / model_filename
            
            shutil.copy2(model_file, registry_path)
            
            # Update metadata
            metadata.model_id = model_id
            metadata.file_path = str(registry_path)
            metadata.file_hash = file_hash
            metadata.file_size = file_size
            metadata.created_at = int(time.time() * 1000)
            
            # Save to database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO models VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                """, (
                    metadata.model_id, metadata.version, metadata.version_type.value,
                    metadata.created_at, metadata.created_by, metadata.description,
                    metadata.model_type, metadata.framework, metadata.file_path,
                    metadata.file_hash, metadata.file_size, metadata.accuracy,
                    metadata.precision, metadata.recall, metadata.f1_score,
                    metadata.sharpe_ratio, metadata.max_drawdown, metadata.win_rate,
                    metadata.profit_factor, metadata.training_data_version,
                    json.dumps(metadata.training_config), json.dumps(metadata.hyperparameters),
                    metadata.deployed, metadata.deployment_env, metadata.deployment_date,
                    json.dumps(metadata.validation_metrics), json.dumps(metadata.backtest_results)
                ))
            
            # Update cache
            self._model_cache[model_id] = metadata
            
            logger.info(f"Model registered: {model_id} v{metadata.version}")
            return True
            
        except Exception as e:
            logger.error(f"Error registering model {model_id}: {e}")
            return False
    
    def register_data(
        self,
        data_id: str,
        data_file: str,
        metadata: DataMetadata
    ) -> bool:
        """
        Register a new data version.
        
        Args:
            data_id: Unique data identifier
            data_file: Path to data file
            metadata: Data metadata
            
        Returns:
            True if registration successful
        """
        try:
            # Validate data file
            if not os.path.exists(data_file):
                logger.error(f"Data file not found: {data_file}")
                return False
            
            # Calculate file hash and size
            file_hash = self._calculate_file_hash(data_file)
            file_size = os.path.getsize(data_file)
            
            # Copy data to registry
            data_filename = f"{data_id}_{metadata.version}.{metadata.file_format}"
            registry_path = self.data_dir / data_filename
            
            shutil.copy2(data_file, registry_path)
            
            # Update metadata
            metadata.data_id = data_id
            metadata.file_path = str(registry_path)
            metadata.file_hash = file_hash
            metadata.file_size = file_size
            metadata.created_at = int(time.time() * 1000)
            
            # Save to database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO data_versions VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                """, (
                    metadata.data_id, metadata.version, metadata.version_type.value,
                    metadata.created_at, metadata.created_by, metadata.description,
                    metadata.data_type, metadata.source, json.dumps(metadata.symbols),
                    metadata.start_date, metadata.end_date, metadata.file_path,
                    metadata.file_hash, metadata.file_size, metadata.file_format,
                    metadata.total_records, metadata.missing_values,
                    metadata.duplicate_records, metadata.quality_score,
                    json.dumps(metadata.preprocessing_steps), json.dumps(metadata.feature_columns),
                    json.dumps(metadata.target_columns)
                ))
            
            # Update cache
            self._data_cache[data_id] = metadata
            
            logger.info(f"Data registered: {data_id} v{metadata.version}")
            return True
            
        except Exception as e:
            logger.error(f"Error registering data {data_id}: {e}")
            return False
    
    def get_model(self, model_id: str, version: Optional[str] = None) -> Optional[ModelMetadata]:
        """
        Get model metadata.
        
        Args:
            model_id: Model identifier
            version: Specific version or None for latest
            
        Returns:
            ModelMetadata or None if not found
        """
        # Check cache first
        if model_id in self._model_cache and version is None:
            return self._model_cache[model_id]
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                if version:
                    cursor = conn.execute("""
                        SELECT * FROM models WHERE model_id = ? AND version = ?
                    """, (model_id, version))
                else:
                    cursor = conn.execute("""
                        SELECT * FROM models WHERE model_id = ? ORDER BY created_at DESC LIMIT 1
                    """, (model_id,))
                
                row = cursor.fetchone()
                if row:
                    metadata = self._row_to_model_metadata(row)
                    if version is None:
                        self._model_cache[model_id] = metadata
                    return metadata
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting model {model_id}: {e}")
            return None
    
    def get_data(self, data_id: str, version: Optional[str] = None) -> Optional[DataMetadata]:
        """
        Get data metadata.
        
        Args:
            data_id: Data identifier
            version: Specific version or None for latest
            
        Returns:
            DataMetadata or None if not found
        """
        # Check cache first
        if data_id in self._data_cache and version is None:
            return self._data_cache[data_id]
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                if version:
                    cursor = conn.execute("""
                        SELECT * FROM data_versions WHERE data_id = ? AND version = ?
                    """, (data_id, version))
                else:
                    cursor = conn.execute("""
                        SELECT * FROM data_versions WHERE data_id = ? ORDER BY created_at DESC LIMIT 1
                    """, (data_id,))
                
                row = cursor.fetchone()
                if row:
                    metadata = self._row_to_data_metadata(row)
                    if version is None:
                        self._data_cache[data_id] = metadata
                    return metadata
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting data {data_id}: {e}")
            return None
    
    def list_models(
        self,
        model_type: Optional[str] = None,
        deployed_only: bool = False,
        limit: int = 100
    ) -> List[ModelMetadata]:
        """
        List models with optional filtering.
        
        Args:
            model_type: Filter by model type
            deployed_only: Only deployed models
            limit: Maximum number of results
            
        Returns:
            List of ModelMetadata
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = "SELECT * FROM models WHERE 1=1"
                params = []
                
                if model_type:
                    query += " AND model_type = ?"
                    params.append(model_type)
                
                if deployed_only:
                    query += " AND deployed = 1"
                
                query += " ORDER BY created_at DESC LIMIT ?"
                params.append(limit)
                
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()
                
                return [self._row_to_model_metadata(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []
    
    def list_data(
        self,
        data_type: Optional[str] = None,
        source: Optional[str] = None,
        limit: int = 100
    ) -> List[DataMetadata]:
        """
        List data versions with optional filtering.
        
        Args:
            data_type: Filter by data type
            source: Filter by source
            limit: Maximum number of results
            
        Returns:
            List of DataMetadata
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = "SELECT * FROM data_versions WHERE 1=1"
                params = []
                
                if data_type:
                    query += " AND data_type = ?"
                    params.append(data_type)
                
                if source:
                    query += " AND source = ?"
                    params.append(source)
                
                query += " ORDER BY created_at DESC LIMIT ?"
                params.append(limit)
                
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()
                
                return [self._row_to_data_metadata(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Error listing data: {e}")
            return []
    
    def deploy_model(
        self,
        model_id: str,
        version: str,
        deployment_env: str = "production"
    ) -> bool:
        """
        Deploy a model version.
        
        Args:
            model_id: Model identifier
            version: Version to deploy
            deployment_env: Deployment environment
            
        Returns:
            True if deployment successful
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Undeploy current version
                conn.execute("""
                    UPDATE models SET deployed = 0, deployment_env = NULL, deployment_date = NULL
                    WHERE model_id = ? AND deployed = 1
                """, (model_id,))
                
                # Deploy new version
                conn.execute("""
                    UPDATE models SET deployed = 1, deployment_env = ?, deployment_date = ?
                    WHERE model_id = ? AND version = ?
                """, (deployment_env, int(time.time() * 1000), model_id, version))
            
            # Update cache
            if model_id in self._model_cache:
                del self._model_cache[model_id]
            
            logger.info(f"Model deployed: {model_id} v{version} to {deployment_env}")
            return True
            
        except Exception as e:
            logger.error(f"Error deploying model {model_id}: {e}")
            return False
    
    def rollback_model(
        self,
        model_id: str,
        target_version: Optional[str] = None
    ) -> bool:
        """
        Rollback model to previous version.
        
        Args:
            model_id: Model identifier
            target_version: Specific version to rollback to (None for previous)
            
        Returns:
            True if rollback successful
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                if target_version:
                    # Rollback to specific version
                    conn.execute("""
                        UPDATE models SET deployed = 1, deployment_env = 'production', deployment_date = ?
                        WHERE model_id = ? AND version = ?
                    """, (int(time.time() * 1000), model_id, target_version))
                else:
                    # Rollback to previous deployed version
                    cursor = conn.execute("""
                        SELECT version FROM models 
                        WHERE model_id = ? AND deployed = 1 
                        ORDER BY deployment_date DESC LIMIT 1 OFFSET 1
                    """, (model_id,))
                    
                    prev_version = cursor.fetchone()
                    if prev_version:
                        prev_version = prev_version[0]
                        
                        # Undeploy current
                        conn.execute("""
                            UPDATE models SET deployed = 0, deployment_env = NULL
                            WHERE model_id = ? AND deployed = 1
                        """, (model_id,))
                        
                        # Deploy previous
                        conn.execute("""
                            UPDATE models SET deployed = 1, deployment_env = 'production', deployment_date = ?
                            WHERE model_id = ? AND version = ?
                        """, (int(time.time() * 1000), model_id, prev_version))
                    else:
                        logger.warning(f"No previous version found for model {model_id}")
                        return False
            
            # Update cache
            if model_id in self._model_cache:
                del self._model_cache[model_id]
            
            logger.info(f"Model rolled back: {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error rolling back model {model_id}: {e}")
            return False
    
    def verify_integrity(self, model_id: Optional[str] = None, data_id: Optional[str] = None) -> Dict[str, bool]:
        """
        Verify file integrity for models and data.
        
        Args:
            model_id: Specific model to verify (None for all)
            data_id: Specific data to verify (None for all)
            
        Returns:
            Dictionary with verification results
        """
        results = {}
        
        # Verify models
        if model_id:
            models_to_check = [model_id]
        else:
            models_to_check = [m.model_id for m in self.list_models(limit=1000)]
        
        for mid in models_to_check:
            metadata = self.get_model(mid)
            if metadata and os.path.exists(metadata.file_path):
                current_hash = self._calculate_file_hash(metadata.file_path)
                results[f"model_{mid}"] = current_hash == metadata.file_hash
            else:
                results[f"model_{mid}"] = False
        
        # Verify data
        if data_id:
            data_to_check = [data_id]
        else:
            data_to_check = [d.data_id for d in self.list_data(limit=1000)]
        
        for did in data_to_check:
            metadata = self.get_data(did)
            if metadata and os.path.exists(metadata.file_path):
                current_hash = self._calculate_file_hash(metadata.file_path)
                results[f"data_{did}"] = current_hash == metadata.file_hash
            else:
                results[f"data_{did}"] = False
        
        return results
    
    def create_version(
        self,
        current_version: str,
        version_type: VersionType
    ) -> str:
        """
        Create next version number.
        
        Args:
            current_version: Current version string
            version_type: Type of version increment
            
        Returns:
            New version string
        """
        try:
            parts = current_version.split('.')
            major = int(parts[0]) if len(parts) > 0 else 0
            minor = int(parts[1]) if len(parts) > 1 else 0
            patch = int(parts[2]) if len(parts) > 2 else 0
            
            if version_type == VersionType.MAJOR:
                major += 1
                minor = 0
                patch = 0
            elif version_type == VersionType.MINOR:
                minor += 1
                patch = 0
            elif version_type == VersionType.PATCH:
                patch += 1
            elif version_type == VersionType.EXPERIMENT:
                patch += 1
            
            return f"{major}.{minor}.{patch}"
            
        except Exception:
            return "1.0.0"
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def _row_to_model_metadata(self, row: tuple) -> ModelMetadata:
        """Convert database row to ModelMetadata."""
        return ModelMetadata(
            model_id=row[0],
            version=row[1],
            version_type=VersionType(row[2]),
            created_at=row[3],
            created_by=row[4],
            description=row[5],
            model_type=row[6],
            framework=row[7],
            file_path=row[8],
            file_hash=row[9],
            file_size=row[10],
            accuracy=row[11],
            precision=row[12],
            recall=row[13],
            f1_score=row[14],
            sharpe_ratio=row[15],
            max_drawdown=row[16],
            win_rate=row[17],
            profit_factor=row[18],
            training_data_version=row[19],
            training_config=json.loads(row[20]) if row[20] else {},
            hyperparameters=json.loads(row[21]) if row[21] else {},
            deployed=bool(row[22]),
            deployment_env=row[23],
            deployment_date=row[24],
            validation_metrics=json.loads(row[25]) if row[25] else {},
            backtest_results=json.loads(row[26]) if row[26] else {}
        )
    
    def _row_to_data_metadata(self, row: tuple) -> DataMetadata:
        """Convert database row to DataMetadata."""
        return DataMetadata(
            data_id=row[0],
            version=row[1],
            version_type=VersionType(row[2]),
            created_at=row[3],
            created_by=row[4],
            description=row[5],
            data_type=row[6],
            source=row[7],
            symbols=json.loads(row[8]) if row[8] else [],
            start_date=row[9],
            end_date=row[10],
            file_path=row[11],
            file_hash=row[12],
            file_size=row[13],
            file_format=row[14],
            total_records=row[15],
            missing_values=row[16],
            duplicate_records=row[17],
            quality_score=row[18],
            preprocessing_steps=json.loads(row[19]) if row[19] else [],
            feature_columns=json.loads(row[20]) if row[20] else [],
            target_columns=json.loads(row[21]) if row[21] else []
        )


# Convenience functions
def create_version_manager(base_path: str = "mlops_registry") -> VersionManager:
    """Create version manager with default settings."""
    return VersionManager(base_path)


def create_model_metadata(
    model_id: str,
    version: str,
    model_type: str,
    framework: str,
    description: str,
    created_by: str = "system"
) -> ModelMetadata:
    """Create model metadata with common fields."""
    return ModelMetadata(
        model_id=model_id,
        version=version,
        version_type=VersionType.MINOR,
        created_at=int(time.time() * 1000),
        created_by=created_by,
        description=description,
        model_type=model_type,
        framework=framework,
        file_path="",
        file_hash="",
        file_size=0
    )


def create_data_metadata(
    data_id: str,
    version: str,
    data_type: str,
    source: str,
    description: str,
    created_by: str = "system"
) -> DataMetadata:
    """Create data metadata with common fields."""
    return DataMetadata(
        data_id=data_id,
        version=version,
        version_type=VersionType.MINOR,
        created_at=int(time.time() * 1000),
        created_by=created_by,
        description=description,
        data_type=data_type,
        source=source,
        file_path="",
        file_hash="",
        file_size=0,
        file_format="parquet"
    )


if __name__ == "__main__":
    # Test version manager
    logging.basicConfig(level=logging.INFO)
    
    vm = create_version_manager("test_registry")
    
    # Create test model metadata
    model_meta = create_model_metadata(
        model_id="test_lstm",
        version="1.0.0",
        model_type="lstm",
        framework="pytorch",
        description="Test LSTM model",
        created_by="developer"
    )
    
    # Add some performance metrics
    model_meta.sharpe_ratio = 1.5
    model_meta.max_drawdown = 0.15
    model_meta.win_rate = 0.65
    model_meta.profit_factor = 2.1
    
    print(f"Created model metadata: {model_meta.model_id}")
    print(f"Version: {model_meta.version}")
    print(f"Sharpe ratio: {model_meta.sharpe_ratio}")
    
    # Test version creation
    new_version = vm.create_version("1.0.0", VersionType.MINOR)
    print(f"Next version: {new_version}")
    
    # Test data metadata
    data_meta = create_data_metadata(
        data_id="market_data_btc",
        version="1.0.0",
        data_type="market_data",
        source="binance",
        description="BTC market data",
        created_by="data_pipeline"
    )
    
    data_meta.symbols = ["BTC/USDT"]
    data_meta.total_records = 1000000
    data_meta.quality_score = 0.95
    
    print(f"Created data metadata: {data_meta.data_id}")
    print(f"Symbols: {data_meta.symbols}")
    print(f"Records: {data_meta.total_records}")
