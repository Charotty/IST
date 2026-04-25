"""
Unit tests for ML and trading metrics.
"""
import pytest
import numpy as np
from its_project.metalearning.metrics import MLMetrics, TradingMetrics


@pytest.mark.unit
@pytest.mark.metalearning_layer
class TestMLMetrics:
    """Test ML metrics functionality."""
    
    def test_accuracy_calculation(self):
        """Test accuracy calculation."""
        y_true = np.array([0, 1, 2, 1, 0])
        y_pred = np.array([0, 1, 2, 0, 0])
        
        metrics = MLMetrics()
        accuracy = metrics.accuracy(y_true, y_pred)
        
        assert accuracy == 0.6  # 3/5 correct
    
    def test_precision_calculation(self):
        """Test precision calculation."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0])
        
        metrics = MLMetrics()
        precision = metrics.precision(y_true, y_pred, average='macro')
        
        assert precision is not None
        assert 0 <= precision <= 1
    
    def test_recall_calculation(self):
        """Test recall calculation."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0])
        
        metrics = MLMetrics()
        recall = metrics.recall(y_true, y_pred, average='macro')
        
        assert recall is not None
        assert 0 <= recall <= 1
    
    def test_f1_score_calculation(self):
        """Test F1 score calculation."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0])
        
        metrics = MLMetrics()
        f1 = metrics.f1_score(y_true, y_pred, average='macro')
        
        assert f1 is not None
        assert 0 <= f1 <= 1
    
    def test_roc_auc_calculation(self):
        """Test ROC AUC calculation."""
        y_true = np.array([0, 1, 0, 1])
        y_proba = np.array([0.1, 0.9, 0.2, 0.8])
        
        metrics = MLMetrics()
        roc_auc = metrics.roc_auc(y_true, y_proba)
        
        assert roc_auc is not None
        assert 0 <= roc_auc <= 1
    
    def test_confusion_matrix(self):
        """Test confusion matrix calculation."""
        y_true = np.array([0, 1, 2, 1, 0])
        y_pred = np.array([0, 1, 2, 0, 0])
        
        metrics = MLMetrics()
        cm = metrics.confusion_matrix(y_true, y_pred)
        
        assert cm is not None
        assert cm.shape == (3, 3)
    
    def test_all_ml_metrics(self):
        """Test calculating all ML metrics."""
        y_true = np.array([0, 1, 2, 1, 0, 2])
        y_pred = np.array([0, 1, 2, 0, 0, 2])
        y_proba = np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.6, 0.3, 0.1],
            [0.7, 0.2, 0.1],
            [0.1, 0.1, 0.8]
        ])
        
        metrics = MLMetrics()
        all_metrics = metrics.calculate_all(y_true, y_pred, y_proba)
        
        assert 'accuracy' in all_metrics
        assert 'precision' in all_metrics
        assert 'recall' in all_metrics
        assert 'f1' in all_metrics
        assert 'roc_auc' in all_metrics
        assert 'confusion_matrix' in all_metrics


@pytest.mark.unit
@pytest.mark.metalearning_layer
class TestTradingMetrics:
    """Test trading metrics functionality."""
    
    def test_returns_calculation(self):
        """Test returns calculation."""
        returns = np.array([0.01, 0.02, -0.01, 0.03, 0.01])
        
        metrics = TradingMetrics()
        total_return = metrics.total_return(returns)
        
        assert total_return is not None
        assert total_return > 0
    
    def test_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        returns = np.array([0.01, 0.02, -0.01, 0.03, 0.01])
        
        metrics = TradingMetrics()
        sharpe = metrics.sharpe_ratio(returns, risk_free_rate=0.02)
        
        assert sharpe is not None
    
    def test_max_drawdown(self):
        """Test maximum drawdown calculation."""
        cumulative = np.array([100, 105, 110, 95, 100, 108])
        
        metrics = TradingMetrics()
        drawdown = metrics.max_drawdown(cumulative)
        
        assert drawdown is not None
        assert drawdown < 0  # Drawdown is negative
    
    def test_win_rate(self):
        """Test win rate calculation."""
        returns = np.array([0.01, 0.02, -0.01, 0.03, -0.02, 0.04])
        
        metrics = TradingMetrics()
        win_rate = metrics.win_rate(returns)
        
        assert win_rate is not None
        assert 0 <= win_rate <= 1
        assert win_rate == 0.5  # 3 wins out of 6
    
    def test_profit_factor(self):
        """Test profit factor calculation."""
        returns = np.array([0.01, 0.02, -0.01, 0.03, -0.02, 0.04])
        
        metrics = TradingMetrics()
        profit_factor = metrics.profit_factor(returns)
        
        assert profit_factor is not None
        assert profit_factor > 0
    
    def test_volatility(self):
        """Test volatility calculation."""
        returns = np.array([0.01, 0.02, -0.01, 0.03, 0.01])
        
        metrics = TradingMetrics()
        vol = metrics.volatility(returns)
        
        assert vol is not None
        assert vol > 0
    
    def test_all_trading_metrics(self):
        """Test calculating all trading metrics."""
        returns = np.array([0.01, 0.02, -0.01, 0.03, -0.02, 0.04, 0.01, -0.01])
        
        metrics = TradingMetrics()
        all_metrics = metrics.calculate_all(returns)
        
        assert 'total_return' in all_metrics
        assert 'sharpe_ratio' in all_metrics
        assert 'max_drawdown' in all_metrics
        assert 'win_rate' in all_metrics
        assert 'profit_factor' in all_metrics
        assert 'volatility' in all_metrics
        assert 'num_trades' in all_metrics
