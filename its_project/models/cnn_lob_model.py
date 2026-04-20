from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from its_project.models.base import BaseModel

logger = logging.getLogger(__name__)


class CNNLOBModel(BaseModel):
    """CNN model specifically designed for order book data."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        
        # Order book specific parameters
        self.depth_levels = config.get("depth_levels", 10)  # Number of price levels
        self.sequence_length = config.get("sequence_length", 20)  # Time steps
        self.num_features = config.get("num_features", 4)  # price, volume, bid/ask
        
        # CNN parameters
        self.conv_channels = config.get("conv_channels", [32, 64, 128])
        self.kernel_sizes = config.get("kernel_sizes", [3, 3, 3])
        self.dropout = config.get("dropout", 0.3)
        self.num_classes = config.get("num_classes", 3)  # sell/hold/buy
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Build model
        self.model = self._build_model()
        self.model.to(self.device)
        
        # Training parameters
        self.epochs = config.get("epochs", 100)
        self.batch_size = config.get("batch_size", 32)
        self.learning_rate = config.get("learning_rate", 0.001)
        self.patience = config.get("patience", 10)
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=config.get("weight_decay", 1e-5)
        )
        self.criterion = nn.CrossEntropyLoss()
        
        self._fitted = False
        self._feature_names = None
    
    def _build_model(self) -> nn.Module:
        """Build CNN model for order book data."""
        class CNNLOBNet(nn.Module):
            def __init__(self, depth_levels: int, sequence_length: int, num_features: int,
                        conv_channels: list, kernel_sizes: list, dropout: float, num_classes: int):
                super().__init__()
                
                # Input shape: (batch, sequence_length, depth_levels, num_features)
                # We'll treat (depth_levels, num_features) as spatial dimensions
                
                self.conv_layers = nn.ModuleList()
                in_channels = num_features
                
                # Build convolutional layers
                for i, (out_ch, kernel_size) in enumerate(zip(conv_channels, kernel_sizes)):
                    self.conv_layers.append(
                        nn.Sequential(
                            nn.Conv2d(in_channels, out_ch, kernel_size, padding=kernel_size//2),
                            nn.BatchNorm2d(out_ch),
                            nn.ReLU(),
                            nn.Dropout2d(dropout)
                        )
                    )
                    in_channels = out_ch
                
                # Calculate flattened size after convolutions
                # After conv layers: (batch, channels, depth_levels, sequence_length)
                self.flattened_size = conv_channels[-1] * depth_levels * sequence_length
                
                # Fully connected layers
                self.fc_layers = nn.Sequential(
                    nn.Linear(self.flattened_size, 512),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(256, num_classes)
                )
                
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # x shape: (batch, sequence_length, depth_levels, num_features)
                batch_size, seq_len, depth, features = x.shape
                
                # Reshape for Conv2d: (batch, features, depth, sequence)
                x = x.permute(0, 3, 2, 1)
                
                # Apply convolutional layers
                for conv_layer in self.conv_layers:
                    x = conv_layer(x)
                
                # Flatten for fully connected layers
                x = x.view(batch_size, -1)
                
                # Apply fully connected layers
                x = self.fc_layers(x)
                
                return x
        
        return CNNLOBNet(
            self.depth_levels,
            self.sequence_length,
            self.num_features,
            self.conv_channels,
            self.kernel_sizes,
            self.dropout,
            self.num_classes
        )
    
    def _prepare_lob_data(self, X: np.ndarray) -> np.ndarray:
        """
        Prepare order book data for CNN.
        
        Expected input format: flatten order book features into (sequence, depth, features)
        """
        if X.ndim == 2:
            # Reshape flat features to (sequence, depth, features)
            # This is a simplified approach - in practice, you'd need proper LOB data structure
            n_samples = X.shape[0]
            # Assume features are organized as [price_level1, volume_level1, price_level2, volume_level2, ...]
            n_features_per_level = 2  # price and volume
            n_levels = self.depth_levels
            
            # Pad or truncate to match expected depth levels
            expected_features = n_levels * n_features_per_level
            if X.shape[1] < expected_features:
                # Pad with zeros
                padding = np.zeros((n_samples, expected_features - X.shape[1]))
                X = np.hstack([X, padding])
            elif X.shape[1] > expected_features:
                # Truncate
                X = X[:, :expected_features]
            
            # Reshape to (samples, sequence_length=1, depth_levels, features)
            X = X.reshape(n_samples, 1, n_levels, n_features_per_level)
        
        elif X.ndim == 3:
            # Already in 3D format, ensure correct dimensions
            n_samples, seq_len, features = X.shape
            if features != self.depth_levels * self.num_features:
                # Adjust features dimension
                if features < self.depth_levels * self.num_features:
                    padding = np.zeros((n_samples, seq_len, self.depth_levels * self.num_features - features))
                    X = np.concatenate([X, padding], axis=2)
                else:
                    X = X[:, :, :self.depth_levels * self.num_features]
            
            # Reshape to include depth dimension
            X = X.reshape(n_samples, seq_len, self.depth_levels, self.num_features)
        
        return X
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "CNNLOBModel":
        """Fit the CNN model."""
        self.validate_input(X, y)
        
        # Prepare LOB data
        X_prepared = self._prepare_lob_data(X)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_prepared).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        
        # Create data loader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Training loop
        best_loss = float('inf')
        patience_counter = 0
        
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            num_batches = 0
            
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                self.best_state_dict = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
        
        # Load best model
        if hasattr(self, 'best_state_dict'):
            self.model.load_state_dict(self.best_state_dict)
        
        self._fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self._fitted:
            raise ValueError("Model must be fitted before prediction")
        
        self.validate_input(X)
        
        X_prepared = self._prepare_lob_data(X)
        X_tensor = torch.FloatTensor(X_prepared).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
        
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self._fitted:
            raise ValueError("Model must be fitted before prediction")
        
        self.validate_input(X)
        
        X_prepared = self._prepare_lob_data(X)
        X_tensor = torch.FloatTensor(X_prepared).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
        
        return probabilities
    
    def get_confidence(self, X: np.ndarray) -> np.ndarray:
        """Get prediction confidence (max probability)."""
        probabilities = self.predict_proba(X)
        return np.max(probabilities, axis=1)
    
    def save(self, path: str) -> None:
        """Save model state."""
        state = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'fitted': self._fitted,
            'feature_names': self._feature_names
        }
        torch.save(state, path)
    
    @classmethod
    def load(cls, path: str) -> "CNNLOBModel":
        """Load model from file."""
        state = torch.load(path, map_location='cpu')
        model = cls(state['config'])
        model.model.load_state_dict(state['model_state_dict'])
        model._fitted = state['fitted']
        model._feature_names = state.get('feature_names')
        return model
    
    @property
    def is_fitted(self) -> bool:
        return self._fitted
