"""
Transformer Model

Transformer нейросеть для временных рядов.
"""

import asyncio
import logging
import time
import math
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import joblib

from ..base.base_model import BaseModel


class TransformerModel(BaseModel):
    """Transformer модель для временных рядов"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Инициализация Transformer модели
        
        Args:
            config: Конфигурация модели
        """
        super().__init__(config)
        
        # Параметры Transformer
        self.input_dim = config.get('input_dim', 50)
        self.d_model = config.get('d_model', 256)
        self.nhead = config.get('nhead', 8)
        self.num_layers = config.get('num_layers', 4)
        self.output_dim = config.get('output_dim', 3)  # BUY, SELL, HOLD
        self.dropout = config.get('dropout', 0.1)
        self.dim_feedforward = config.get('dim_feedforward', 512)
        
        # Параметры обучения
        self.learning_rate = config.get('learning_rate', 0.0001)
        self.batch_size = config.get('batch_size', 32)
        self.epochs = config.get('epochs', 100)
        self.patience = config.get('patience', 10)
        self.validation_split = config.get('validation_split', 0.2)
        
        # Параметры последовательности
        self.sequence_length = config.get('sequence_length', 60)
        
        # Инициализация модели
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        # Построение модели
        self._build_model()
        
        # Оптимизатор и функция потерь
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
        # История обучения
        self.training_history = []
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
    
    def _build_model(self) -> None:
        """Построение Transformer модели"""
        try:
            self.model = self._TransformerNetwork(
                input_dim=self.input_dim,
                d_model=self.d_model,
                nhead=self.nhead,
                num_layers=self.num_layers,
                dim_feedforward=self.dim_feedforward,
                output_dim=self.output_dim,
                dropout=self.dropout,
                sequence_length=self.sequence_length
            ).to(self.device)
            
            self.model_info = {
                'architecture': 'Transformer',
                'parameters': self._count_parameters(),
                'input_dim': self.input_dim,
                'd_model': self.d_model,
                'nhead': self.nhead,
                'num_layers': self.num_layers,
                'output_dim': self.output_dim,
                'dim_feedforward': self.dim_feedforward
            }
            
            self.logger.info(f"Transformer model built with {self._count_parameters()} parameters")
            
        except Exception as e:
            self._handle_error(e, "_build_model")
            raise
    
    class _PositionalEncoding(nn.Module):
        """Позиционное кодирование"""
        
        def __init__(self, d_model: int, max_len: int = 5000):
            super().__init__()
            
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            
            div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                               (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term[0::2])
            pe[:, 1::2] = torch.cos(position * div_term[1::2])
            
            self.register_buffer('pe', pe.unsqueeze(0).transpose(0, 1))
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.pe[:, :x.size(1)]
    
    class _TransformerNetwork(nn.Module):
        """Внутренний класс Transformer сети"""
        
        def __init__(self, input_dim: int, d_model: int, nhead: int, num_layers: int,
                     dim_feedforward: int, output_dim: int, dropout: float, sequence_length: int):
            super().__init__()
            
            self.d_model = d_model
            self.sequence_length = sequence_length
            
            # Проекция входа в d_model
            self.input_projection = nn.Linear(input_dim, d_model)
            
            # Позиционное кодирование
            self.pos_encoder = TransformerModel._PositionalEncoding(d_model, sequence_length)
            
            # Transformer encoder
            self.transformer_encoder = nn.Transformer(
                d_model=d_model,
                nhead=nhead,
                num_encoder_layers=num_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True
            )
            
            # Выходная проекция
            self.output_projection = nn.Linear(d_model, output_dim)
            self.dropout = nn.Dropout(dropout)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x shape: (batch_size, seq_len, input_dim)
            
            # Проекция входа
            x = self.input_projection(x)  # (batch_size, seq_len, d_model)
            
            # Позиционное кодирование
            x = self.pos_encoder(x)  # (batch_size, seq_len, d_model)
            
            # Transformer encoder
            # Transformer ожидает (seq_len, batch_size, d_model)
            x = x.transpose(0, 1)  # (seq_len, batch_size, d_model)
            
            transformer_out = self.transformer_encoder(x)  # (seq_len, batch_size, d_model)
            
            # Возврат к (batch_size, seq_len, d_model)
            transformer_out = transformer_out.transpose(0, 1)
            
            # Используем последний выход
            last_output = transformer_out[:, -1, :]  # (batch_size, d_model)
            
            # Dropout и выходной слой
            output = self.dropout(last_output)
            output = self.output_projection(output)  # (batch_size, output_dim)
            
            return output
    
    def _count_parameters(self) -> int:
        """Подсчет параметров модели"""
        return sum(p.numel() for p in self.model.parameters())
    
    def _create_sequences(self, data: np.ndarray, target: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Создание последовательностей для Transformer
        
        Args:
            data: Входные данные
            target: Целевая переменная
            
        Returns:
            Tuple: Последовательности и цели
        """
        sequences = []
        targets = []
        
        for i in range(len(data) - self.sequence_length):
            sequences.append(data[i:i + self.sequence_length])
            if target is not None:
                targets.append(target[i + self.sequence_length])
        
        return np.array(sequences), np.array(targets) if target is not None else None
    
    def _prepare_tensors(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Подготовка тензоров
        
        Args:
            X: Входные данные
            y: Целевая переменная
            
        Returns:
            Tuple: Тензоры
        """
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        if y is not None:
            y_tensor = torch.LongTensor(y).to(self.device)
            return X_tensor, y_tensor
        else:
            return X_tensor, None
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> None:
        """
        Обучение Transformer модели
        
        Args:
            X: Признаки
            y: Целевая переменная
            **kwargs: Дополнительные параметры
        """
        try:
            start_time = time.time()
            
            # Подготовка данных
            X_clean, y_clean = self.prepare_data(X, y)
            
            # Масштабирование
            X_scaled = self.scaler_X.fit_transform(X_clean)
            y_scaled = self.scaler_y.fit_transform(y_clean.values.reshape(-1, 1)).flatten()
            
            # Создание последовательностей
            X_seq, y_seq = self._create_sequences(X_scaled, y_scaled)
            
            if len(X_seq) == 0:
                raise ValueError("Not enough data to create sequences")
            
            # Разделение на train/validation
            split_idx = int(len(X_seq) * (1 - self.validation_split))
            X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
            y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]
            
            # Подготовка тензоров
            X_train_tensor, y_train_tensor = self._prepare_tensors(X_train, y_train)
            X_val_tensor, y_val_tensor = self._prepare_tensors(X_val, y_val)
            
            # Создание DataLoader
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
            
            self._log_training_start(X_train_tensor.shape)
            
            # Цикл обучения
            for epoch in range(self.epochs):
                # Training
                self.model.train()
                train_loss = 0.0
                
                for batch_X, batch_y in train_loader:
                    self.optimizer.zero_grad()
                    
                    outputs = self.model(batch_X)
                    loss = self.criterion(outputs, batch_y)
                    
                    loss.backward()
                    self.optimizer.step()
                    
                    train_loss += loss.item()
                
                # Validation
                self.model.eval()
                val_loss = 0.0
                correct = 0
                total = 0
                
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        outputs = self.model(batch_X)
                        loss = self.criterion(outputs, batch_y)
                        val_loss += loss.item()
                        
                        _, predicted = torch.max(outputs.data, 1)
                        total += batch_y.size(0)
                        correct += (predicted == batch_y).sum().item()
                
                avg_train_loss = train_loss / len(train_loader)
                avg_val_loss = val_loss / len(val_loader)
                val_accuracy = correct / total if total > 0 else 0
                
                # Сохранение истории
                self.training_history.append({
                    'epoch': epoch + 1,
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'val_accuracy': val_accuracy
                })
                
                # Early stopping
                if avg_val_loss < self.best_val_loss:
                    self.best_val_loss = avg_val_loss
                    self.epochs_without_improvement = 0
                    # Сохранение лучшей модели
                    torch.save(self.model.state_dict(), 'best_transformer_model.pth')
                else:
                    self.epochs_without_improvement += 1
                
                if self.epochs_without_improvement >= self.patience:
                    self.logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
                
                # Логирование
                if (epoch + 1) % 10 == 0:
                    self.logger.info(f"Epoch {epoch + 1}/{self.epochs}: "
                                  f"Train Loss: {avg_train_loss:.4f}, "
                                  f"Val Loss: {avg_val_loss:.4f}, "
                                  f"Val Acc: {val_accuracy:.4f}")
            
            # Загрузка лучшей модели
            if os.path.exists('best_transformer_model.pth'):
                self.model.load_state_dict(torch.load('best_transformer_model.pth'))
                os.remove('best_transformer_model.pth')
            
            self.is_trained = True
            training_time = time.time() - start_time
            self._log_training_end(training_time)
            
        except Exception as e:
            self._handle_error(e, "fit")
            raise
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Предсказание
        
        Args:
            X: Признаки
            
        Returns:
            np.ndarray: Предсказания
        """
        try:
            start_time = time.time()
            
            if not self.is_trained:
                raise ValueError("Model must be trained before prediction")
            
            # Подготовка данных
            X_clean, _ = self.prepare_data(X)
            X_scaled = self.scaler_X.transform(X_clean)
            
            # Создание последовательностей
            X_seq, _ = self._create_sequences(X_scaled)
            
            if len(X_seq) == 0:
                return np.array([])
            
            # Подготовка тензоров
            X_tensor, _ = self._prepare_tensors(X_seq)
            
            # Предсказание
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(X_tensor)
                _, predicted = torch.max(outputs.data, 1)
            
            predictions = predicted.cpu().numpy()
            
            prediction_time = time.time() - start_time
            self._log_prediction(len(predictions), prediction_time)
            
            return predictions
            
        except Exception as e:
            self._handle_error(e, "predict")
            return np.array([])
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Предсказание вероятностей
        
        Args:
            X: Признаки
            
        Returns:
            np.ndarray: Вероятности
        """
        try:
            start_time = time.time()
            
            if not self.is_trained:
                raise ValueError("Model must be trained before prediction")
            
            # Подготовка данных
            X_clean, _ = self.prepare_data(X)
            X_scaled = self.scaler_X.transform(X_clean)
            
            # Создание последовательностей
            X_seq, _ = self._create_sequences(X_scaled)
            
            if len(X_seq) == 0:
                return np.array([])
            
            # Подготовка тензоров
            X_tensor, _ = self._prepare_tensors(X_seq)
            
            # Предсказание вероятностей
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(X_tensor)
                probabilities = torch.softmax(outputs, dim=1)
            
            probs = probabilities.cpu().numpy()
            
            prediction_time = time.time() - start_time
            self._log_prediction(len(probs), prediction_time)
            
            return probs
            
        except Exception as e:
            self._handle_error(e, "predict_proba")
            return np.array([])
    
    def save_model(self, path: str) -> None:
        """
        Сохранение модели
        
        Args:
            path: Путь для сохранения
        """
        try:
            # Создание директории если необходимо
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Сохранение модели
            model_data = {
                'model_state_dict': self.model.state_dict(),
                'scaler_X': self.scaler_X,
                'scaler_y': self.scaler_y,
                'config': self.config,
                'training_history': self.training_history
            }
            
            torch.save(model_data, path)
            
            # Сохранение метаданных
            self._save_metadata(path)
            
            self.logger.info(f"Transformer model saved to {path}")
            
        except Exception as e:
            self._handle_error(e, "save_model")
            raise
    
    def load_model(self, path: str) -> None:
        """
        Загрузка модели
        
        Args:
            path: Путь к модели
        """
        try:
            # Загрузка модели
            model_data = torch.load(path, map_location=self.device)
            
            self.model.load_state_dict(model_data['model_state_dict'])
            self.scaler_X = model_data['scaler_X']
            self.scaler_y = model_data['scaler_y']
            self.config = model_data['config']
            self.training_history = model_data.get('training_history', [])
            
            # Обновление параметров
            self.input_dim = self.config.get('input_dim', 50)
            self.d_model = self.config.get('d_model', 256)
            self.nhead = self.config.get('nhead', 8)
            self.num_layers = self.config.get('num_layers', 4)
            self.output_dim = self.config.get('output_dim', 3)
            self.dropout = self.config.get('dropout', 0.1)
            self.dim_feedforward = self.config.get('dim_feedforward', 512)
            
            self.is_trained = True
            
            # Загрузка метаданных
            self._load_metadata(path)
            
            self.logger.info(f"Transformer model loaded from {path}")
            
        except Exception as e:
            self._handle_error(e, "load_model")
            raise
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Transformer не предоставляет важность признаков
        
        Returns:
            Dict: Пустой словарь
        """
        return None
    
    def get_training_history(self) -> List[Dict[str, Any]]:
        """
        Получение истории обучения
        
        Returns:
            List: История обучения
        """
        return self.training_history.copy()
