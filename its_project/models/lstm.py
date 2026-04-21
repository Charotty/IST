from __future__ import annotations

from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from its_project.models.base import BaseModel
except ImportError:
    from .base import BaseModel


class LSTMModel(BaseModel):
    """LSTM for direction prediction (PyTorch)."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.input_size = config.get("input_size", 50)
        self.hidden_size = config.get("hidden_size", 128)
        self.num_layers = config.get("num_layers", 2)
        self.num_classes = config.get("num_classes", 3)  # sell, hold, buy
        self.dropout = config.get("dropout", 0.2)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model()

    def _build_model(self) -> nn.Module:
        class LSTMNet(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout,
                )
                self.fc = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size // 2, num_classes),
                )

            def forward(self, x):
                # x: (batch, seq_len, features)
                lstm_out, _ = self.lstm(x)
                last_out = lstm_out[:, -1, :]
                return self.fc(last_out)

        net = LSTMNet(
            self.input_size, self.hidden_size, self.num_layers, self.num_classes, self.dropout
        )
        return net.to(self.device)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LSTMModel":
        """
        Train LSTM.

        Args:
            X: shape (n_samples, seq_len, n_features)
            y: shape (n_samples,)
        """
        self.validate_input(X)
        if X.ndim != 3:
            raise ValueError(f"LSTM expects 3D input, got {X.ndim}D")
        # Store feature count (last dim)
        self.feature_names = [f"feat_{i}" for i in range(X.shape[2])]

        # Prepare tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.get("batch_size", 32),
            shuffle=True,
        )

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.get("learning_rate", 0.001),
        )
        criterion = nn.CrossEntropyLoss()

        epochs = self.config.get("epochs", 100)
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(dataloader)
                # Use logger if available; otherwise print
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        self.validate_input(X)
        if X.ndim != 3:
            raise ValueError(f"LSTM expects 3D input, got {X.ndim}D")

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
            probs = torch.softmax(outputs, dim=1)
        return probs.cpu().numpy()
