"""
Model Registry Module

Model registry for managing model versions, metadata, and lifecycle.
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, List
import hashlib


class ModelRegistry:
    """
    Registry for managing model versions and metadata.
    """
    
    def __init__(self, registry_path='model_registry.json'):
        """
        Initialize model registry.
        
        :param registry_path: Path to registry file
        """
        self.registry_path = registry_path
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
    
    def register_model(self, model_name: str, model: Any, metadata: Dict[str, Any]) -> str:
        """
        Register a model in the registry.
        
        :param model_name: Name of the model
        :param model: Model instance
        :param metadata: Model metadata
        :return: Model ID
        """
        model_id = self._generate_model_id(model_name)
        
        entry = {
            'model_id': model_id,
            'model_name': model_name,
            'registered_at': datetime.now().isoformat(),
            'metadata': metadata,
            'status': 'active'
        }
        
        self.registry['models'][model_id] = entry
        self._save_registry()
        
        return model_id
    
    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Get model entry by ID.
        
        :param model_id: Model ID
        :return: Model entry dict or None
        """
        return self.registry['models'].get(model_id)
    
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
    
    def get_active_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get the active model for a given name.
        
        :param model_name: Model name
        :return: Active model entry or None
        """
        models = self.list_models(model_name)
        active_models = [m for m in models if m['status'] == 'active']
        return active_models[0] if active_models else None
