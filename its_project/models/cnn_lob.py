"""
CNN Model for Limit Order Book (LOB) Data Processing
Based on VKR requirements for DeepLOB-style architecture
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional


class CNNLOBModel(nn.Module):
    """
    CNN model for processing Limit Order Book data.
    
    Based on DeepLOB architecture (Zhang et al.) which uses CNN
    to extract spatial features from LOB structure.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize CNN-LOB model.
        
        Args:
            config: Dictionary with model parameters:
                - input_channels: Number of input channels (e.g., price, volume per level)
                - num_levels: Number of LOB levels
                - seq_len: Sequence length
                - num_classes: Number of output classes
                - dropout: Dropout rate
        """
        super(CNNLOBModel, self).__init__()
        
        self.input_channels = config.get('input_channels', 40)  # 20 levels * 2 (price, volume)
        self.num_levels = config.get('num_levels', 20)
        self.seq_len = config.get('seq_len', 100)
        self.num_classes = config.get('num_classes', 3)
        self.dropout = config.get('dropout', 0.2)
        
        # CNN layers for spatial feature extraction (from LOB levels)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(1, 2), stride=(1, 2))
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(1, 2), stride=(1, 2))
        self.conv3 = nn.Conv2d(32, 32, kernel_size=(1, 10), stride=(1, 10))
        
        # LSTM for temporal dependencies
        self.lstm = nn.LSTM(
            input_size=32 * (self.num_levels // 40),  # Adjust based on conv output
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=self.dropout
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, self.num_classes)
        
        # Dropout and activation
        self.dropout_layer = nn.Dropout(self.dropout)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_channels)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        batch_size, seq_len, channels = x.shape
        
        # Reshape for 2D CNN: (batch, 1, seq_len, channels)
        x = x.unsqueeze(1)
        
        # CNN layers for spatial feature extraction
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        
        # Reshape for LSTM: (batch, seq_len, features)
        x = x.squeeze(1).permute(0, 2, 1)
        
        # LSTM for temporal dependencies
        lstm_out, _ = self.lstm(x)
        
        # Take last time step
        last_out = lstm_out[:, -1, :]
        
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
            X_train: Training features of shape (n_samples, seq_len, input_channels)
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
