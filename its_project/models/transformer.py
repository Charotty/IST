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


class TransformerModel(BaseModel):
    """Transformer for sequence modeling (PyTorch)."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.input_size = config.get("input_size", 50)
        self.d_model = config.get("d_model", 128)
        self.nhead = config.get("nhead", 8)
        self.num_layers = config.get("num_layers", 4)
        self.num_classes = config.get("num_classes", 3)
        self.dropout = config.get("dropout", 0.1)
        self.max_seq_len = config.get("max_seq_len", 200)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model()

    def _build_model(self) -> nn.Module:
        class TransformerNet(nn.Module):
            def __init__(self, input_size, d_model, nhead, num_layers, num_classes, max_seq_len, dropout):
                super().__init__()
                # Input projection
                self.input_proj = nn.Linear(input_size, d_model)
                # Positional encoding
                self.pos_encoder = nn.Parameter(torch.randn(1, max_seq_len, d_model))
                # Transformer encoder
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=d_model * 4,
                    dropout=dropout,
                    batch_first=True,
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                # Classification head
                self.fc = nn.Linear(d_model, num_classes)

            def forward(self, x):
                # x: (batch, seq_len, features)
                batch_size, seq_len, _ = x.shape
                # Project to d_model
                x = self.input_proj(x)
                # Add positional encoding (truncate/pad if needed)
                pos_enc = self.pos_encoder[:, :seq_len, :]
                x = x + pos_enc
                # Transformer
                x = self.transformer(x)
                # Global average pooling
                x = x.mean(dim=1)
                # Classification
                return self.fc(x)

        net = TransformerNet(
            self.input_size,
            self.d_model,
            self.nhead,
            self.num_layers,
            self.num_classes,
            self.max_seq_len,
            self.dropout,
        )
        return net.to(self.device)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TransformerModel":
        """Train Transformer."""
        self.validate_input(X)
        if X.ndim != 3:
            raise ValueError(f"Transformer expects 3D input, got {X.ndim}D")
        self.feature_names = [f"feat_{i}" for i in range(X.shape[2])]

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
            raise ValueError(f"Transformer expects 3D input, got {X.ndim}D")

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
            probs = torch.softmax(outputs, dim=1)
        return probs.cpu().numpy()
