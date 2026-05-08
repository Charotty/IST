#!/usr/bin/env python3
"""
Configuration Management System
==============================

Production-ready configuration management:
- Hierarchical configuration
- Environment-specific configs
- Validation and type checking
- Hot reloading
- Configuration versioning
- Secret management
"""

from __future__ import annotations

import os
import json
import yaml
import logging
from typing import Dict, Any, Optional, List, Union, Type, get_type_hints
from dataclasses import dataclass, field, asdict
from pathlib import Path
from enum import Enum
import threading
from datetime import datetime
import hashlib
import inspect
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class Environment(Enum):
    """Environment types."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class ConfigFormat(Enum):
    """Configuration file formats."""
    JSON = "json"
    YAML = "yaml"
    TOML = "toml"
    ENV = "env"


@dataclass
class ConfigSource:
    """Configuration source definition."""
    name: str
    path: str
    format: ConfigFormat
    priority: int = 0
    required: bool = True
    watch_for_changes: bool = False
    secrets: List[str] = field(default_factory=list)


@dataclass
class ConfigValidation:
    """Configuration validation rule."""
    field_path: str
    field_type: Type
    required: bool = True
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    allowed_values: Optional[List[Any]] = None
    pattern: Optional[str] = None
    custom_validator: Optional[callable] = None


class ConfigManager:
    """
    Production-ready configuration management system.
    
    Features:
    - Hierarchical configuration loading
    - Environment-specific overrides
    - Type validation and checking
    - Hot reloading
    - Secret management
    - Configuration versioning
    """
    
    def __init__(
        self,
        app_name: str,
        environment: Optional[Environment] = None,
        config_dir: str = "config"
    ) -> None:
        self.app_name = app_name
        self.environment = environment or self._detect_environment()
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration storage
        self._config: Dict[str, Any] = {}
        self._sources: List[ConfigSource] = []
        self._validations: List[ConfigValidation] = []
        
        # Hot reloading
        self._watchers: Dict[str, threading.Thread] = {}
        self._watch_running = True
        self._callbacks: List[callable] = []
        
        # Secret management
        self._secrets: Dict[str, str] = {}
        self._secret_sources: List[str] = []
        
        # Version tracking
        self._config_version: str = "1.0.0"
        self._last_modified: Dict[str, float] = {}
        
        # Lock for thread safety
        self._lock = threading.RLock()
        
        # Initialize default sources
        self._setup_default_sources()
    
    def _detect_environment(self) -> Environment:
        """Detect current environment."""
        env_var = os.environ.get('ENVIRONMENT', '').lower()
        
        if env_var == 'production' or env_var == 'prod':
            return Environment.PRODUCTION
        elif env_var == 'staging':
            return Environment.STAGING
        elif env_var == 'testing' or env_var == 'test':
            return Environment.TESTING
        else:
            return Environment.DEVELOPMENT
    
    def _setup_default_sources(self) -> None:
        """Setup default configuration sources."""
        # Base configuration
        self.add_source(ConfigSource(
            name="base",
            path=f"{self.app_name}.yaml",
            format=ConfigFormat.YAML,
            priority=100,
            required=True
        ))
        
        # Environment-specific configuration
        self.add_source(ConfigSource(
            name="environment",
            path=f"{self.app_name}.{self.environment.value}.yaml",
            format=ConfigFormat.YAML,
            priority=200,
            required=False
        ))
        
        # Local overrides
        self.add_source(ConfigSource(
            name="local",
            path=f"{self.app_name}.local.yaml",
            format=ConfigFormat.YAML,
            priority=300,
            required=False,
            watch_for_changes=True
        ))
        
        # Environment variables
        self.add_source(ConfigSource(
            name="env_vars",
            path=".env",
            format=ConfigFormat.ENV,
            priority=400,
            required=False
        ))
    
    def add_source(self, source: ConfigSource) -> None:
        """Add configuration source."""
        with self._lock:
            self._sources.append(source)
            self._sources.sort(key=lambda x: x.priority)
    
    def add_validation(self, validation: ConfigValidation) -> None:
        """Add configuration validation rule."""
        with self._lock:
            self._validations.append(validation)
    
    def add_callback(self, callback: callable) -> None:
        """Add configuration change callback."""
        self._callbacks.append(callback)
    
    def load_config(self) -> Dict[str, Any]:
        """Load and merge all configuration sources."""
        with self._lock:
            merged_config = {}
            
            for source in self._sources:
                try:
                    config_data = self._load_source(source)
                    if config_data:
                        merged_config = self._merge_configs(merged_config, config_data)
                        self._last_modified[source.name] = time.time()
                        logger.debug(f"Loaded config from {source.name}")
                except Exception as e:
                    if source.required:
                        logger.error(f"Required config source {source.name} failed: {e}")
                        raise
                    else:
                        logger.warning(f"Optional config source {source.name} failed: {e}")
            
            # Apply environment variable overrides
            env_overrides = self._load_env_overrides()
            merged_config = self._merge_configs(merged_config, env_overrides)
            
            # Validate configuration
            self._validate_config(merged_config)
            
            # Store configuration
            old_config = self._config.copy()
            self._config = merged_config
            
            # Trigger callbacks if config changed
            if old_config != merged_config:
                self._trigger_callbacks(old_config, merged_config)
            
            # Start file watchers
            self._start_watchers()
            
            logger.info(f"Configuration loaded for {self.app_name} in {self.environment.value}")
            return merged_config
    
    def _load_source(self, source: ConfigSource) -> Optional[Dict[str, Any]]:
        """Load configuration from source."""
        file_path = self.config_dir / source.path
        
        if not file_path.exists():
            return None
        
        try:
            if source.format == ConfigFormat.YAML:
                with open(file_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
            elif source.format == ConfigFormat.JSON:
                with open(file_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            elif source.format == ConfigFormat.TOML:
                try:
                    import toml
                    with open(file_path, 'r', encoding='utf-8') as f:
                        config = toml.load(f)
                except ImportError:
                    logger.error("TOML support requires 'toml' package")
                    return None
            elif source.format == ConfigFormat.ENV:
                config = self._load_env_file(file_path)
            else:
                raise ValueError(f"Unsupported format: {source.format}")
            
            # Process secrets
            if source.secrets:
                config = self._process_secrets(config, source.secrets)
            
            return config or {}
            
        except Exception as e:
            logger.error(f"Error loading config from {file_path}: {e}")
            raise
    
    def _load_env_file(self, file_path: Path) -> Dict[str, Any]:
        """Load environment variables from file."""
        config = {}
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"\'')
                    
                    # Try to parse as JSON, otherwise keep as string
                    try:
                        config[key] = json.loads(value)
                    except:
                        config[key] = value
        
        return config
    
    def _load_env_overrides(self) -> Dict[str, Any]:
        """Load environment variable overrides."""
        overrides = {}
        prefix = f"{self.app_name.upper()}_"
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower().replace('_', '.')
                
                # Try to parse as JSON, otherwise keep as string
                try:
                    overrides[config_key] = json.loads(value)
                except:
                    overrides[config_key] = value
        
        return overrides
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two configuration dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if '.' in key:
                # Nested key
                keys = key.split('.')
                current = result
                
                for k in keys[:-1]:
                    if k not in current:
                        current[k] = {}
                    current = current[k]
                
                current[keys[-1]] = value
            else:
                # Top-level key
                if isinstance(value, dict) and isinstance(result.get(key), dict):
                    result[key] = self._merge_configs(result[key], value)
                else:
                    result[key] = value
        
        return result
    
    def _process_secrets(self, config: Dict[str, Any], secret_keys: List[str]) -> Dict[str, Any]:
        """Process secret configuration values."""
        for key in secret_keys:
            if key in config:
                secret_value = config[key]
                if isinstance(secret_value, str) and secret_value.startswith('${') and secret_value.endswith('}'):
                    # Environment variable reference
                    env_var = secret_value[2:-1]
                    config[key] = os.environ.get(env_var, '')
                else:
                    # Store in secrets
                    self._secrets[key] = str(secret_value)
        
        return config
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration against rules."""
        for validation in self._validations:
            try:
                value = self._get_nested_value(config, validation.field_path)
                
                if value is None:
                    if validation.required:
                        raise ValueError(f"Required field {validation.field_path} is missing")
                    continue
                
                # Type validation
                if not isinstance(value, validation.field_type):
                    try:
                        value = validation.field_type(value)
                        self._set_nested_value(config, validation.field_path, value)
                    except (ValueError, TypeError):
                        raise ValueError(f"Field {validation.field_path} must be of type {validation.field_type.__name__}")
                
                # Range validation
                if validation.min_value is not None and value < validation.min_value:
                    raise ValueError(f"Field {validation.field_path} must be >= {validation.min_value}")
                
                if validation.max_value is not None and value > validation.max_value:
                    raise ValueError(f"Field {validation.field_path} must be <= {validation.max_value}")
                
                # Allowed values validation
                if validation.allowed_values and value not in validation.allowed_values:
                    raise ValueError(f"Field {validation.field_path} must be one of {validation.allowed_values}")
                
                # Pattern validation
                if validation.pattern and isinstance(value, str):
                    import re
                    if not re.match(validation.pattern, value):
                        raise ValueError(f"Field {validation.field_path} does not match pattern {validation.pattern}")
                
                # Custom validation
                if validation.custom_validator:
                    if not validation.custom_validator(value):
                        raise ValueError(f"Field {validation.field_path} failed custom validation")
                
            except Exception as e:
                logger.error(f"Configuration validation failed: {e}")
                raise
    
    def _get_nested_value(self, config: Dict[str, Any], path: str) -> Any:
        """Get nested value from config."""
        keys = path.split('.')
        current = config
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        
        return current
    
    def _set_nested_value(self, config: Dict[str, Any], path: str, value: Any) -> None:
        """Set nested value in config."""
        keys = path.split('.')
        current = config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def _trigger_callbacks(self, old_config: Dict[str, Any], new_config: Dict[str, Any]) -> None:
        """Trigger configuration change callbacks."""
        for callback in self._callbacks:
            try:
                callback(old_config, new_config)
            except Exception as e:
                logger.error(f"Error in config callback: {e}")
    
    def _start_watchers(self) -> None:
        """Start file watchers for hot reloading."""
        for source in self._sources:
            if source.watch_for_changes:
                if source.name not in self._watchers:
                    watcher = threading.Thread(
                        target=self._watch_file,
                        args=(source,),
                        daemon=True
                    )
                    watcher.start()
                    self._watchers[source.name] = watcher
    
    def _watch_file(self, source: ConfigSource) -> None:
        """Watch configuration file for changes."""
        file_path = self.config_dir / source.path
        
        if not file_path.exists():
            return
        
        last_modified = file_path.stat().st_mtime
        
        while self._watch_running:
            try:
                current_modified = file_path.stat().st_mtime
                
                if current_modified > last_modified:
                    logger.info(f"Configuration file {source.path} changed, reloading...")
                    self.load_config()
                    last_modified = current_modified
                
                threading.Event().wait(1.0)  # Check every second
                
            except Exception as e:
                logger.error(f"Error watching file {source.path}: {e}")
                threading.Event().wait(5.0)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        with self._lock:
            return self._get_nested_value(self._config, key) or default
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        with self._lock:
            self._set_nested_value(self._config, key, value)
    
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration."""
        with self._lock:
            return self._config.copy()
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get configuration section."""
        section_config = self.get(section, {})
        return section_config if isinstance(section_config, dict) else {}
    
    def reload(self) -> Dict[str, Any]:
        """Reload configuration."""
        return self.load_config()
    
    def save_config(self, file_path: Optional[str] = None) -> None:
        """Save current configuration to file."""
        if file_path is None:
            file_path = f"{self.app_name}_saved.yaml"
        
        save_path = self.config_dir / file_path
        
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(self._config, f, default_flow_style=False, indent=2)
        
        logger.info(f"Configuration saved to {save_path}")
    
    def get_config_hash(self) -> str:
        """Get hash of current configuration."""
        config_str = json.dumps(self._config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def get_version(self) -> str:
        """Get configuration version."""
        return self._config_version
    
    def set_version(self, version: str) -> None:
        """Set configuration version."""
        self._config_version = version
    
    def get_environment(self) -> Environment:
        """Get current environment."""
        return self.environment
    
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == Environment.PRODUCTION
    
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.environment == Environment.DEVELOPMENT
    
    def get_secret(self, key: str) -> Optional[str]:
        """Get secret value."""
        return self._secrets.get(key)
    
    def add_secret(self, key: str, value: str) -> None:
        """Add secret value."""
        self._secrets[key] = value
    
    def shutdown(self) -> None:
        """Shutdown configuration manager."""
        self._watch_running = False
        
        # Wait for watchers to finish
        for watcher in self._watchers.values():
            if watcher.is_alive():
                watcher.join(timeout=1)
        
        logger.info("Configuration manager shutdown")


class ConfigBuilder:
    """Builder for configuration classes."""
    
    @staticmethod
    def from_dataclass(cls: Type, config: Dict[str, Any]) -> Type:
        """Create dataclass instance from configuration dict."""
        # Get type hints
        hints = get_type_hints(cls)
        
        # Filter and convert values
        filtered_config = {}
        for key, value in config.items():
            if key in hints:
                target_type = hints[key]
                try:
                    if target_type == bool and isinstance(value, str):
                        filtered_config[key] = value.lower() in ('true', '1', 'yes', 'on')
                    else:
                        filtered_config[key] = target_type(value)
                except (ValueError, TypeError):
                    logger.warning(f"Could not convert {key}={value} to {target_type}")
        
        return cls(**filtered_config)


# Convenience functions
def create_config_manager(
    app_name: str,
    environment: Optional[Environment] = None,
    config_dir: str = "config"
) -> ConfigManager:
    """Create configuration manager with default settings."""
    return ConfigManager(app_name, environment, config_dir)


def create_production_config_manager(app_name: str) -> ConfigManager:
    """Create production-ready configuration manager."""
    manager = ConfigManager(
        app_name=app_name,
        environment=Environment.PRODUCTION,
        config_dir="/etc/config" if os.path.exists("/etc/config") else "config"
    )
    
    # Add common validations
    manager.add_validation(ConfigValidation(
        field_path="database.host",
        field_type=str,
        required=True,
        pattern=r'^[a-zA-Z0-9.-]+$'
    ))
    
    manager.add_validation(ConfigValidation(
        field_path="database.port",
        field_type=int,
        required=True,
        min_value=1,
        max_value=65535
    ))
    
    manager.add_validation(ConfigValidation(
        field_path="api.rate_limit",
        field_type=int,
        required=False,
        min_value=1,
        max_value=10000
    ))
    
    return manager


if __name__ == "__main__":
    # Test configuration manager
    logging.basicConfig(level=logging.INFO)
    
    # Create test config directory
    test_config_dir = Path("test_config")
    test_config_dir.mkdir(exist_ok=True)
    
    # Create test configuration files
    base_config = {
        "database": {
            "host": "localhost",
            "port": 5432,
            "name": "test_db"
        },
        "api": {
            "rate_limit": 100,
            "timeout": 30
        },
        "logging": {
            "level": "INFO",
            "file": "app.log"
        }
    }
    
    with open(test_config_dir / "test_app.yaml", 'w') as f:
        yaml.dump(base_config, f)
    
    env_config = {
        "database": {
            "host": "prod-server"
        },
        "api": {
            "rate_limit": 1000
        }
    }
    
    with open(test_config_dir / "test_app.production.yaml", 'w') as f:
        yaml.dump(env_config, f)
    
    # Test configuration manager
    manager = create_config_manager("test_app", Environment.PRODUCTION, "test_config")
    
    # Add validation
    manager.add_validation(ConfigValidation(
        field_path="database.port",
        field_type=int,
        required=True,
        min_value=1,
        max_value=65535
    ))
    
    # Load configuration
    config = manager.load_config()
    print(f"Loaded config: {json.dumps(config, indent=2)}")
    
    # Test getters
    print(f"Database host: {manager.get('database.host')}")
    print(f"API rate limit: {manager.get('api.rate_limit')}")
    print(f"Environment: {manager.get_environment()}")
    print(f"Is production: {manager.is_production()}")
    
    # Test section
    db_config = manager.get_section("database")
    print(f"Database config: {db_config}")
    
    # Cleanup
    import shutil
    shutil.rmtree(test_config_dir)
    
    print("Configuration system ready!")
