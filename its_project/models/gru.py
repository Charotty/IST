"""
GRU Model for Time Series Prediction
Based on VKR requirements for cryptocurrency price forecasting
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional


class GRUModel(nn.Module):
    """
    Gated Recurrent Unit (GRU) model for time series prediction.
    
    According to VKR, GRU is more efficient than LSTM for crypto markets
    due to fewer gates and faster convergence on volatile data.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize GRU model.
        
        Args:
            config: Dictionary with model parameters:
                - input_size: Number of input features
                - hidden_size: Size of hidden state
                - num_layers: Number of GRU layers
                - output_size: Number of output classes (for classification)
                - dropout: Dropout rate
                - bidirectional: Whether to use bidirectional GRU
        """
        super(GRUModel, self).__init__()
        
        self.input_size = config.get('input_size', 10)
        self.hidden_size = config.get('hidden_size', 64)
        self.num_layers = config.get('num_layers', 2)
        self.output_size = config.get('output_size', 3)  # 3 classes: SELL, HOLD, BUY
        self.dropout = config.get('dropout', 0.2)
        self.bidirectional = config.get('bidirectional', False)
        
        # GRU layer
        self.gru = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=self.bidirectional,
            batch_first=True
        )
        
        # Calculate final hidden size
        gru_output_size = self.hidden_size * 2 if self.bidirectional else self.hidden_size
        
        # Fully connected layers
        self.fc1 = nn.Linear(gru_output_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, self.output_size)
        
        # Dropout
        self.dropout_layer = nn.Dropout(self.dropout)
        
        # Activation
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # GRU forward pass
        gru_out, _ = self.gru(x)
        
        # Take last time step output
        if self.bidirectional:
            # Concatenate forward and backward outputs
            last_out = torch.cat((gru_out[:, -1, :self.hidden_size], 
                                 gru_out[:, 0, self.hidden_size:]), dim=1)
        else:
            last_out = gru_out[:, -1, :]
        
        # Fully connected layers
        out = self.dropout_layer(last_out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout_layer(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        
        return out
    
    def predict_proba(self, x: torch.Tensor) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            x: Input tensor
            
        Returns:
            Array of class probabilities
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = self.softmax(logits)
        return probs.cpu().numpy()
    
    def predict(self, x: torch.Tensor) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            x: Input tensor
            
        Returns:
            Array of predicted class labels
        """
        probs = self.predict_proba(x)
        return np.argmax(probs, axis=1)
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
            epochs: int = 100, batch_size: int = 32, learning_rate: float = 0.001):
        """
        Train the model.
        
        Args:
            X_train: Training features of shape (n_samples, seq_len, input_size)
            y_train: Training labels of shape (n_samples,)
            X_val: Validation features
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
        """
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        
        # Create data loader
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Validation data
        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val)
            y_val_tensor = torch.LongTensor(y_val)
            val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        # Training loop
        self.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.forward(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            
            # Validation
            if val_loader is not None and (epoch + 1) % 10 == 0:
                self.eval()
                val_loss = 0
                correct = 0
                total = 0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        outputs = self.forward(batch_X)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        total += batch_y.size(0)
                        correct += (predicted == batch_y).sum().item()
                
                val_accuracy = 100 * correct / total
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}, "
                      f"Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_accuracy:.2f}%")
                self.train()
            elif (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}")
