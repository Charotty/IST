"""
Siamese Architecture for Bid/Ask LOB Analysis
Based on VKR requirements for modeling symmetry between demand and supply
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Tuple


class SiameseEncoder(nn.Module):
    """
    Shared encoder for processing bid and ask sides of LOB.
    Uses identical parameters for both sides to capture symmetric patterns.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Siamese encoder.
        
        Args:
            config: Dictionary with model parameters:
                - input_size: Number of input features per level
                - hidden_size: Size of hidden layers
                - num_layers: Number of LSTM layers
        """
        super(SiameseEncoder, self).__init__()
        
        self.input_size = config.get('input_size', 2)  # price, volume per level
        self.hidden_size = config.get('hidden_size', 64)
        self.num_layers = config.get('num_layers', 2)
        self.dropout = config.get('dropout', 0.2)
        
        # LSTM for temporal processing
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.hidden_size, 128)
        self.fc2 = nn.Linear(128, 64)
        
        # Dropout and activation
        self.dropout_layer = nn.Dropout(self.dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            Encoded tensor of shape (batch_size, 64)
        """
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # Take last time step
        last_out = lstm_out[:, -1, :]
        
        # Fully connected
        out = self.dropout_layer(last_out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout_layer(out)
        out = self.fc2(out)
        
        return out


class SiameseLOBModel(nn.Module):
    """
    Siamese model for processing bid and ask sides of LOB.
    
    Uses two identical encoders with shared parameters to process
    bid and ask data separately, then combines the outputs.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Siamese LOB model.
        
        Args:
            config: Dictionary with model parameters:
                - input_size: Number of input features per level
                - hidden_size: Size of hidden layers
                - num_layers: Number of LSTM layers
                - num_classes: Number of output classes
                - dropout: Dropout rate
        """
        super(SiameseLOBModel, self).__init__()
        
        self.num_classes = config.get('num_classes', 3)
        
        # Shared encoder for bid and ask
        self.encoder = SiameseEncoder(config)
        
        # Combination layer (merge bid and ask encodings)
        self.combination = nn.Sequential(
            nn.Linear(128, 64),  # 64 (bid) + 64 (ask) = 128
            nn.ReLU(),
            nn.Dropout(config.get('dropout', 0.2)),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(config.get('dropout', 0.2))
        )
        
        # Classification head
        self.classifier = nn.Linear(32, self.num_classes)
        
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, bid_data: torch.Tensor, ask_data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            bid_data: Bid side LOB data of shape (batch_size, seq_len, input_size)
            ask_data: Ask side LOB data of shape (batch_size, seq_len, input_size)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Encode bid and ask separately with shared encoder
        bid_encoded = self.encoder(bid_data)
        ask_encoded = self.encoder(ask_data)
        
        # Combine encodings
        combined = torch.cat([bid_encoded, ask_encoded], dim=1)
        
        # Process combination
        out = self.combination(combined)
        
        # Classify
        logits = self.classifier(out)
        
        return logits
    
    def predict_proba(self, bid_data: torch.Tensor, ask_data: torch.Tensor) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            bid_data: Bid side LOB data
            ask_data: Ask side LOB data
            
        Returns:
            Array of class probabilities
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(bid_data, ask_data)
            probs = self.softmax(logits)
        return probs.cpu().numpy()
    
    def predict(self, bid_data: torch.Tensor, ask_data: torch.Tensor) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            bid_data: Bid side LOB data
            ask_data: Ask side LOB data
            
        Returns:
            Array of predicted class labels
        """
        probs = self.predict_proba(bid_data, ask_data)
        return np.argmax(probs, axis=1)
    
    def fit(self, bid_train: np.ndarray, ask_train: np.ndarray, y_train: np.ndarray,
            bid_val: Optional[np.ndarray] = None, ask_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            epochs: int = 100, batch_size: int = 32, learning_rate: float = 0.001):
        """
        Train the model.
        
        Args:
            bid_train: Training bid data of shape (n_samples, seq_len, input_size)
            ask_train: Training ask data of shape (n_samples, seq_len, input_size)
            y_train: Training labels of shape (n_samples,)
            bid_val: Validation bid data
            ask_val: Validation ask data
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
        """
        # Convert to tensors
        bid_train_tensor = torch.FloatTensor(bid_train)
        ask_train_tensor = torch.FloatTensor(ask_train)
        y_train_tensor = torch.LongTensor(y_train)
        
        # Create data loader
        train_dataset = torch.utils.data.TensorDataset(bid_train_tensor, ask_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Validation data
        val_loader = None
        if bid_val is not None and ask_val is not None and y_val is not None:
            bid_val_tensor = torch.FloatTensor(bid_val)
            ask_val_tensor = torch.FloatTensor(ask_val)
            y_val_tensor = torch.LongTensor(y_val)
            val_dataset = torch.utils.data.TensorDataset(bid_val_tensor, ask_val_tensor, y_val_tensor)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        # Training loop
        self.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_bid, batch_ask, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.forward(batch_bid, batch_ask)
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
                    for batch_bid, batch_ask, batch_y in val_loader:
                        outputs = self.forward(batch_bid, batch_ask)
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
