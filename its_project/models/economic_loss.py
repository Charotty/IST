from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional


class EconomicLoss(nn.Module):
    """
    Economic loss function that optimizes for trading profitability.
    
    Instead of standard cross-entropy, this loss considers:
    - Expected returns of each action
    - Transaction costs
    - Risk-adjusted returns
    - Class imbalance
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        
        # Economic parameters
        self.expected_returns = config.get("expected_returns", [0.01, 0.0, -0.01])  # BUY, HOLD, SELL
        self.transaction_costs = config.get("transaction_costs", 0.001)  # 0.1% per trade
        self.risk_aversion = config.get("risk_aversion", 1.0)
        
        # Loss components weights
        self.pnl_weight = config.get("pnl_weight", 1.0)
        self.classification_weight = config.get("classification_weight", 0.5)
        self.regularization_weight = config.get("regularization_weight", 0.1)
        
        # Class weights for imbalance
        self.class_weights = config.get("class_weights", None)
        if self.class_weights is None:
            self.class_weights = torch.tensor([1.0, 1.0, 1.0])
        else:
            self.class_weights = torch.tensor(self.class_weights)
        
        # Base loss
        self.ce_loss = nn.CrossEntropyLoss(weight=self.class_weights, reduction='none')
        
        # Risk metrics
        self.use_sharpe = config.get("use_sharpe", True)
        self.sharpe_window = config.get("sharpe_window", 20)
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate economic loss.
        
        Args:
            logits: Model predictions (batch_size, n_classes)
            targets: True labels (batch_size,)
            
        Returns:
            Economic loss tensor
        """
        # Standard classification loss
        ce_loss = self.ce_loss(logits, targets)
        
        # Economic PnL loss
        pnl_loss = self._calculate_pnl_loss(logits, targets)
        
        # Risk-adjusted loss
        if self.use_sharpe:
            sharpe_loss = self._calculate_sharpe_loss(logits, targets)
        else:
            sharpe_loss = torch.tensor(0.0)
        
        # Regularization
        reg_loss = self._calculate_regularization_loss(logits)
        
        # Combine losses
        total_loss = (
            self.classification_weight * ce_loss.mean() +
            self.pnl_weight * pnl_loss +
            self.regularization_weight * reg_loss +
            sharpe_loss
        )
        
        return total_loss
    
    def _calculate_pnl_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate PnL-based loss."""
        # Get probabilities
        probs = torch.softmax(logits, dim=-1)
        
        # Calculate expected PnL for each prediction
        expected_pnl = torch.zeros_like(probs[:, 0])
        
        for i, (prob, true_class) in enumerate(zip(probs, targets)):
            # Expected return for predicted action
            pred_returns = torch.tensor(self.expected_returns, dtype=torch.float32)
            expected_return = (prob * pred_returns).sum()
            
            # Transaction cost for non-HOLD actions
            transaction_cost = 0.0
            if prob[0] > 0.1 or prob[2] > 0.1:  # SELL or BUY with significant probability
                transaction_cost = self.transaction_costs
            
            # True action return (what we should have gotten)
            true_return = self.expected_returns[true_class.item()]
            
            # Opportunity cost: difference between optimal and expected
            opportunity_cost = true_return - expected_return
            
            # PnL loss: maximize expected return minus costs
            pnl = expected_return - transaction_cost - self.risk_aversion * torch.abs(opportunity_cost)
            expected_pnl[i] = -pnl  # Negative because we minimize loss
        
        return expected_pnl.mean()
    
    def _calculate_sharpe_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate Sharpe ratio-based loss."""
        if logits.size(0) < self.sharpe_window:
            return torch.tensor(0.0)
        
        probs = torch.softmax(logits, dim=-1)
        
        # Calculate returns for each sample
        returns = []
        for prob in probs:
            pred_returns = torch.tensor(self.expected_returns, dtype=torch.float32)
            expected_return = (prob * pred_returns).sum()
            returns.append(expected_return)
        
        returns = torch.stack(returns)
        
        # Calculate Sharpe ratio
        mean_return = returns.mean()
        std_return = returns.std()
        
        if std_return > 1e-8:
            sharpe = mean_return / std_return
            sharpe_loss = -sharpe  # Maximize Sharpe ratio
        else:
            sharpe_loss = torch.tensor(0.0)
        
        return sharpe_loss
    
    def _calculate_regularization_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """Calculate regularization loss."""
        # L2 regularization on logits
        l2_loss = torch.norm(logits, p=2)
        
        # Entropy regularization (encourage confident predictions)
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
        entropy_loss = -entropy.mean()  # Minimize negative entropy (maximize entropy)
        
        return l2_loss + 0.1 * entropy_loss


class AsymmetricLoss(nn.Module):
    """
    Asymmetric loss function that penalizes wrong trading directions more heavily.
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        
        # Asymmetric penalties
        self.buy_sell_penalty = config.get("buy_sell_penalty", 2.0)  # Penalty for wrong direction
        self.buy_hold_penalty = config.get("buy_hold_penalty", 1.5)  # Penalty for missed opportunity
        self.sell_hold_penalty = config.get("sell_hold_penalty", 1.5)
        
        # Base loss
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate asymmetric loss."""
        ce_loss = self.ce_loss(logits, targets)
        
        # Get predictions
        predictions = torch.argmax(logits, dim=-1)
        
        # Apply asymmetric penalties
        weights = torch.ones_like(targets, dtype=torch.float32)
        
        for i, (pred, true) in enumerate(zip(predictions, targets)):
            if true == 2 and pred == 0:  # True BUY, predicted SELL
                weights[i] = self.buy_sell_penalty
            elif true == 0 and pred == 2:  # True SELL, predicted BUY
                weights[i] = self.buy_sell_penalty
            elif true == 2 and pred == 1:  # True BUY, predicted HOLD
                weights[i] = self.buy_hold_penalty
            elif true == 0 and pred == 1:  # True SELL, predicted HOLD
                weights[i] = self.sell_hold_penalty
            # HOLD predictions have no special penalty
        
        weighted_loss = ce_loss * weights
        return weighted_loss.mean()


class ProfitWeightedLoss(nn.Module):
    """
    Loss function weighted by potential profit of each class.
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        
        # Profit weights for each class
        self.profit_weights = config.get("profit_weights", [1.0, 0.1, 1.0])  # SELL, HOLD, BUY
        
        # Convert to tensor
        self.weights = torch.tensor(self.profit_weights)
        
        # Base loss
        self.ce_loss = nn.CrossEntropyLoss(weight=self.weights, reduction='mean')
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate profit-weighted loss."""
        return self.ce_loss(logits, targets)


class RiskAdjustedLoss(nn.Module):
    """
    Risk-adjusted loss function considering volatility and drawdown.
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        
        # Risk parameters
        self.var_penalty = config.get("var_penalty", 0.5)
        self.drawdown_penalty = config.get("drawdown_penalty", 0.3)
        
        # Base loss
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate risk-adjusted loss."""
        ce_loss = self.ce_loss(logits, targets)
        
        probs = torch.softmax(logits, dim=-1)
        
        # Calculate risk metrics
        var_penalty = self._calculate_var_penalty(probs)
        drawdown_penalty = self._calculate_drawdown_penalty(probs)
        
        total_loss = ce_loss.mean() + self.var_penalty * var_penalty + self.drawdown_penalty * drawdown_penalty
        return total_loss
    
    def _calculate_var_penalty(self, probs: torch.Tensor) -> torch.Tensor:
        """Calculate Value at Risk penalty."""
        # Simplified VAR calculation based on prediction uncertainty
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
        var_penalty = torch.var(entropy)
        return var_penalty
    
    def _calculate_drawdown_penalty(self, probs: torch.Tensor) -> torch.Tensor:
        """Calculate drawdown penalty."""
        # Simplified drawdown based on probability changes
        if probs.size(0) < 2:
            return torch.tensor(0.0)
        
        prob_changes = torch.diff(probs, dim=0)
        max_drawdown = torch.max(torch.cumsum(torch.abs(prob_changes), dim=0))
        return torch.mean(max_drawdown)
