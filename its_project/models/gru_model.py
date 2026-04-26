from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from its_project.models.base import BaseModel
except ImportError:
    from .base import BaseModel

logger = logging.getLogger(__name__)


class GRUModel(BaseModel):
    """GRU-based model for time series prediction."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.input_size = config.get("input_size", 50)
        self.hidden_size = config.get("hidden_size", 128)
        self.num_layers = config.get("num_layers", 2)
        self.dropout = config.get("dropout", 0.2)
        self.bidirectional = config.get("bidirectional", False)
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
        """Build GRU model architecture."""
        class GRUNet(nn.Module):
            def __init__(self, input_size: int, hidden_size: int, num_layers: int, 
                       dropout: float, bidirectional: bool, num_classes: int):
                super().__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                self.bidirectional = bidirectional
                
                self.gru = nn.GRU(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0,
                    bidirectional=bidirectional
                )
                
                # Adjust output size for bidirectional
                gru_output_size = hidden_size * (2 if bidirectional else 1)
                
                self.dropout = nn.Dropout(dropout)
                self.fc = nn.Linear(gru_output_size, hidden_size // 2)
                self.relu = nn.ReLU()
                self.output = nn.Linear(hidden_size // 2, num_classes)
                
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # x shape: (batch_size, seq_len, input_size)
                batch_size = x.size(0)
                
                # Initialize hidden state
                h0 = torch.zeros(
                    self.num_layers * (2 if self.bidirectional else 1),
                    batch_size,
                    self.hidden_size,
                    device=x.device
                )
                
                # GRU forward
                out, _ = self.gru(x, h0)
                
                # Use last time step
                if self.bidirectional:
                    # Concatenate forward and backward last states
                    out = torch.cat((out[:, -1, :self.hidden_size], out[:, 0, self.hidden_size:]), dim=1)
                else:
                    out = out[:, -1, :]
                
                # Fully connected layers
                out = self.dropout(out)
                out = self.relu(self.fc(out))
                out = self.output(out)
                
                return out
        
        return GRUNet(
            self.input_size,
            self.hidden_size,
            self.num_layers,
            self.dropout,
            self.bidirectional,
            self.num_classes
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray, progress_callback=None) -> "GRUModel":
        """Fit the GRU model."""
        self.validate_input(X, y)
        
        # Convert to tensors
        if X.ndim == 2:
            # Add sequence dimension if needed
            X = X.reshape(X.shape[0], 1, X.shape[1])
        
        X_tensor = torch.FloatTensor(X).to(self.device)
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
            
            # Call progress callback if provided
            if progress_callback:
                progress_callback(epoch + 1, self.epochs, avg_loss)
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                # Save best model
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
        
        self._is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        self.validate_input(X)
        
        if X.ndim == 2:
            X = X.reshape(X.shape[0], 1, X.shape[1])
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        
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
        
        if X.ndim == 2:
            X = X.reshape(X.shape[0], 1, X.shape[1])
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        
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
            'fitted': self._is_fitted,
            'feature_names': self._feature_names
        }
        torch.save(state, path)
    
    @classmethod
    def load(cls, path: str) -> "GRUModel":
        """Load model from file."""
        state = torch.load(path, map_location='cpu')
        model = cls(state['config'])
        model.model.load_state_dict(state['model_state_dict'])
        model._is_fitted = state.get('fitted', False)
        model._feature_names = state.get('feature_names', None)
        return model
