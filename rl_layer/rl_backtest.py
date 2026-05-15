"""
RL Backtesting utilities for evaluating risk management strategies.

Provides functions to run backtests with RL-based dynamic risk selection
and compare against static risk baselines.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
import matplotlib.pyplot as plt

from .dqn_agent import DQNAgent
from .environment import TradingEnvironment
from .ensemble_environment import EnsembleTradingEnv
from .microstructure_environment import MicrostructureRLenv


def run_rl_backtest(
    df: pd.DataFrame,
    agent: DQNAgent,
    env_class,
    episodes: int = 5,
    verbose: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run RL backtest with trained agent.
    
    For each bar: action -> risk_multiplier
    strategy_return = final_signal * pct_change.shift(-1) * risk_multiplier
    
    Args:
        df: DataFrame with OHLCV data and features
        agent: Trained DQN agent
        env_class: Environment class to use (TradingEnvironment, EnsembleTradingEnv, etc.)
        episodes: Number of episodes to run
        verbose: Whether to print progress
        
    Returns:
        Tuple of (results_df, metrics_dict)
    """
    # Create environment
    env = env_class(df)
    
    # Store results
    all_results = []
    
    for episode in range(episodes):
        if verbose:
            print(f"Episode {episode + 1}/{episodes}")
        
        state = env.reset()
        episode_rewards = []
        episode_actions = []
        episode_risk_multipliers = []
        
        done = False
        step = 0
        
        while not done:
            # Get action from agent (no exploration during backtest)
            action = agent.act(state, training=False)
            risk_multiplier = agent.get_risk_multiplier(action)
            
            # Take step
            next_state, reward, done, info = env.step(action)
            
            # Store results
            episode_rewards.append(reward)
            episode_actions.append(action)
            episode_risk_multipliers.append(risk_multiplier)
            
            state = next_state
            step += 1
        
        # Calculate episode metrics
        episode_df = pd.DataFrame({
            'step': range(len(episode_rewards)),
            'reward': episode_rewards,
            'action': episode_actions,
            'risk_multiplier': episode_risk_multipliers,
            'episode': episode
        })
        
        all_results.append(episode_df)
    
    # Combine all episodes
    results_df = pd.concat(all_results, ignore_index=True)
    
    # Calculate metrics
    metrics = {
        'total_episodes': episodes,
        'avg_reward': results_df['reward'].mean(),
        'std_reward': results_df['reward'].std(),
        'total_reward': results_df['reward'].sum(),
        'avg_risk_multiplier': results_df['risk_multiplier'].mean(),
    }
    
    if verbose:
        print(f"\nBacktest Results:")
        print(f"Average Reward: {metrics['avg_reward']:.4f}")
        print(f"Total Reward: {metrics['total_reward']:.4f}")
        print(f"Average Risk Multiplier: {metrics['avg_risk_multiplier']:.4f}")
    
    return results_df, metrics


def run_static_backtest(
    df: pd.DataFrame,
    static_risk: float = 0.01
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run backtest with static risk multiplier (baseline).
    
    Args:
        df: DataFrame with OHLCV data and features
        static_risk: Static risk multiplier (e.g., 0.01 for 1%)
        
    Returns:
        Tuple of (results_df, metrics_dict)
    """
    # Calculate price changes
    df = df.copy()
    df['pct_change'] = df['close'].pct_change()
    df['future_pct_change'] = df['pct_change'].shift(-1)
    
    # Calculate strategy returns
    df['strategy_return'] = df['final_signal'] * df['future_pct_change'] * static_risk * 100
    
    # Remove NaN values
    df = df.dropna(subset=['strategy_return'])
    
    # Calculate metrics
    metrics = {
        'static_risk': static_risk,
        'total_return': df['strategy_return'].sum(),
        'avg_return': df['strategy_return'].mean(),
        'std_return': df['strategy_return'].std(),
        'sharpe_ratio': df['strategy_return'].mean() / df['strategy_return'].std() if df['strategy_return'].std() > 0 else 0,
        'total_trades': len(df)
    }
    
    results_df = df[['strategy_return', 'final_signal', 'future_pct_change']].copy()
    
    return results_df, metrics


def compare_rl_vs_static(
    df: pd.DataFrame,
    agent: DQNAgent,
    env_class,
    static_risk: float = 0.01,
    episodes: int = 5
) -> Dict[str, Any]:
    """
    Compare RL dynamic risk vs static risk baseline.
    
    Args:
        df: DataFrame with OHLCV data and features
        agent: Trained DQN agent
        env_class: Environment class to use
        static_risk: Static risk multiplier for baseline
        episodes: Number of episodes for RL backtest
        
    Returns:
        Dictionary with comparison metrics
    """
    # Run RL backtest
    rl_results, rl_metrics = run_rl_backtest(df, agent, env_class, episodes, verbose=False)
    
    # Run static backtest
    static_results, static_metrics = run_static_backtest(df, static_risk)
    
    # Comparison
    comparison = {
        'rl': rl_metrics,
        'static': static_metrics,
        'improvement': {
            'total_reward': rl_metrics['total_reward'] - static_metrics['total_return'],
            'avg_reward': rl_metrics['avg_reward'] - static_metrics['avg_return'],
            'improvement_pct': (rl_metrics['total_reward'] - static_metrics['total_return']) / abs(static_metrics['total_return']) * 100 if static_metrics['total_return'] != 0 else 0
        }
    }
    
    return comparison


def train_agent(
    df: pd.DataFrame,
    env_class,
    episodes: int = 5,
    steps_per_episode: int = 500,
    target_update_freq: int = 10,
    verbose: bool = True
) -> DQNAgent:
    """
    Train DQN agent on given environment.
    
    Args:
        df: DataFrame with OHLCV data and features
        env_class: Environment class to use
        episodes: Number of training episodes
        steps_per_episode: Maximum steps per episode
        target_update_freq: Frequency of target network updates
        verbose: Whether to print progress
        
    Returns:
        Trained DQN agent
    """
    # Create environment
    env = env_class(df)
    
    # Create agent
    agent = DQNAgent(state_size=env.state_size, action_size=env.action_size)
    
    episode_rewards = []
    
    for episode in range(episodes):
        if verbose:
            print(f"Training Episode {episode + 1}/{episodes}")
        
        state = env.reset()
        total_reward = 0
        step = 0
        
        while step < steps_per_episode:
            # Choose action
            action = agent.act(state, training=True)
            
            # Take step
            next_state, reward, done, info = env.step(action)
            
            # Store experience
            agent.remember(state, action, reward, next_state, done)
            
            # Train agent
            loss = agent.replay()
            
            total_reward += reward
            state = next_state
            step += 1
            
            if done:
                break
        
        # Update target network
        if episode % target_update_freq == 0:
            agent.update_target_model()
        
        episode_rewards.append(total_reward)
        
        if verbose:
            print(f"Episode {episode + 1} - Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}")
    
    if verbose:
        print(f"\nTraining completed. Average reward: {np.mean(episode_rewards):.2f}")
    
    return agent


def plot_comparison(rl_results: pd.DataFrame, static_results: pd.DataFrame, save_path: str = None):
    """
    Plot comparison between RL and static risk strategies.
    
    Args:
        rl_results: Results DataFrame from RL backtest
        static_results: Results DataFrame from static backtest
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Cumulative returns
    axes[0].plot(rl_results['reward'].cumsum(), label='RL Dynamic Risk', alpha=0.7)
    axes[0].plot(static_results['strategy_return'].cumsum(), label='Static Risk (1%)', alpha=0.7)
    axes[0].set_title('Cumulative Returns Comparison')
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Cumulative Return')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Risk multiplier distribution
    axes[1].hist(rl_results['risk_multiplier'], bins=30, alpha=0.7, label='RL Risk Multiplier')
    axes[1].axvline(0.01, color='red', linestyle='--', label='Static Risk (1%)')
    axes[1].set_title('Risk Multiplier Distribution')
    axes[1].set_xlabel('Risk Multiplier')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
