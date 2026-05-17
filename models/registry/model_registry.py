"""
Model Registry Module

Model registry for managing model versions, metadata, and lifecycle.
"""

import json
import os
import pickle
from datetime import datetime
from typing import Dict, Any, Optional, List
import hashlib


class ModelRegistry:
    """
    Registry for managing model versions and metadata.
    """
    
    def __init__(self, registry_path='model_registry.json', models_dir='models/'):
        """
        Initialize model registry.
        
        :param registry_path: Path to registry file
        :param models_dir: Directory for saving model weights
        """
        self.registry_path = registry_path
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        self.registry = self._load_registry()
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load registry from file."""
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, 'r') as f:
                    return json.load(f)
            except:
                return {'models': {}}
        return {'models': {}}
    
    def _save_registry(self):
        """Save registry to file."""
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=4)
    
    def _generate_model_id(self, model_name: str) -> str:
        """Generate unique model ID."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"{model_name}_{timestamp}"
    
    def _compute_feature_hash(self, feature_cols: List[str]) -> str:
        """Compute hash of feature columns for schema versioning."""
        feature_str = ','.join(sorted(feature_cols))
        return hashlib.sha256(feature_str.encode()).hexdigest()[:16]
    
    def register_model(self, model_name: str, model: Any, metadata: Dict[str, Any], 
                      feature_cols: Optional[List[str]] = None, save_weights: bool = True,
                      artifact_manifest: Optional[Dict[str, Any]] = None) -> str:
        """
        Register a model in the registry.
        
        :param model_name: Name of the model
        :param model: Model instance
        :param metadata: Model metadata
        :param feature_cols: List of feature columns used for training
        :param save_weights: Whether to save model weights
        :param artifact_manifest: Optional reproducibility bundle (seeds, hyperparams, data fingerprints, paths)
        :return: Model ID
        """
        model_id = self._generate_model_id(model_name)
        
        # Compute schema version from feature columns
        schema_version = self._compute_feature_hash(feature_cols) if feature_cols else 'unknown'
        
        # Save model weights if requested
        model_path = None
        if save_weights:
            model_path = os.path.join(self.models_dir, f"{model_id}.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        
        entry = {
            'model_id': model_id,
            'model_name': model_name,
            'registered_at': datetime.now().isoformat(),
            'metadata': metadata,
            'schema_version': schema_version,
            'feature_cols': feature_cols,
            'model_path': model_path,
            'status': 'active'
        }
        if artifact_manifest is not None:
            entry['artifact_manifest'] = artifact_manifest
        
        self.registry['models'][model_id] = entry
        self._save_registry()
        
        return model_id
    
    def get_model(self, model_id: str, load_weights: bool = False) -> Optional[Dict[str, Any]]:
        """
        Get model entry by ID.
        
        :param model_id: Model ID
        :param load_weights: Whether to load model weights
        :return: Model entry dict or None
        """
        entry = self.registry['models'].get(model_id)
        if entry and load_weights and entry.get('model_path'):
            try:
                with open(entry['model_path'], 'rb') as f:
                    entry['model_instance'] = pickle.load(f)
            except Exception as e:
                print(f"Warning: Failed to load model weights: {e}")
        return entry
    
    def list_models(self, model_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all models or models by name.
        
        :param model_name: Optional model name filter
        :return: List of model entries
        """
        models = list(self.registry['models'].values())
        
        if model_name:
            models = [m for m in models if m['model_name'] == model_name]
        
        return sorted(models, key=lambda x: x['registered_at'], reverse=True)
    
    def deactivate_model(self, model_id: str):
        """
        Deactivate a model.
        
        :param model_id: Model ID
        """
        if model_id in self.registry['models']:
            self.registry['models'][model_id]['status'] = 'inactive'
            self.registry['models'][model_id]['deactivated_at'] = datetime.now().isoformat()
            self._save_registry()
    
    def get_active_model(self, model_name: str, load_weights: bool = False) -> Optional[Dict[str, Any]]:
        """
        Get the active model for a given name.
        
        :param model_name: Model name
        :param load_weights: Whether to load model weights
        :return: Active model entry or None
        """
        models = self.list_models(model_name)
        active_models = [m for m in models if m['status'] == 'active']
        if active_models:
            entry = active_models[0]
            if load_weights and entry.get('model_path'):
                try:
                    with open(entry['model_path'], 'rb') as f:
                        entry['model_instance'] = pickle.load(f)
                except Exception as e:
                    print(f"Warning: Failed to load model weights: {e}")
            return entry
        return None
    
    def validate_schema(self, model_id: str, feature_cols: List[str]) -> bool:
        """
        Validate that model schema matches current feature set.
        
        :param model_id: Model ID
        :param feature_cols: Current feature columns
        :return: True if schema matches
        """
        entry = self.registry['models'].get(model_id)
        if not entry:
            return False
        
        current_hash = self._compute_feature_hash(feature_cols)
        return entry.get('schema_version') == current_hash
