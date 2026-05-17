"""
Reinforcement Learning Layer for ITS

This layer provides RL-based risk management on top of the final signal from Meta-Learning.
It does NOT predict market direction - it chooses the risk level (position sizing).

Components:
- TradingEnvironment: Basic RL environment with 10-dimensional state
- EnsembleTradingEnv: Environment using ensemble probability instead of direction probability
- MicrostructureRLenv: Extended environment with order book imbalance (OBI)
- DQNAgent: Deep Q-Network agent for risk level selection
- RL backtesting utilities
"""

from .environment import TradingEnvironment
from .ensemble_environment import EnsembleTradingEnv
from .microstructure_environment import MicrostructureRLenv
from .dqn_agent import DQNAgent
from .rl_backtest import run_rl_backtest, run_rl_overlay_backtest
from .integration import prepare_df_for_rl_env, prepare_from_training_result

__all__ = [
    'TradingEnvironment',
    'EnsembleTradingEnv',
    'MicrostructureRLenv',
    'DQNAgent',
    'run_rl_backtest',
    'run_rl_overlay_backtest',
    'prepare_df_for_rl_env',
    'prepare_from_training_result',
]
