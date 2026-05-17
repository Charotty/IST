"""
RL Backtesting utilities for evaluating risk management strategies.

Оценка через тот же ``backtesting.Backtester``, что и основной контур (исполнение на
следующем баре, комиссии при смене эффективной экспозиции).

Cf. Gymnasium custom environments: наблюдение, пошаговый ``step``, ``reset``
(https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

from .dqn_agent import DQNAgent


def run_rl_backtest(
    df: pd.DataFrame,
    agent: DQNAgent,
    env_class,
    episodes: int = 5,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Несколько эпизодов прохода по истории (как в обучении DQN).

    Для сопоставимости с prod-PnL см. ``run_rl_overlay_backtest`` (один проход + Backtester).
    """
    env = env_class(df)
    all_results = []

    for episode in range(episodes):
        if verbose:
            print(f"Episode {episode + 1}/{episodes}")

        state = env.reset()
        episode_rewards = []
        episode_actions = []
        episode_risk_multipliers = []

        done = False
        while not done:
            action = agent.act(state, training=False)
            risk_multiplier = agent.get_risk_multiplier(action)
            next_state, reward, done, info = env.step(action)
            episode_rewards.append(reward)
            episode_actions.append(action)
            episode_risk_multipliers.append(risk_multiplier)
            state = next_state

        episode_df = pd.DataFrame({
            "step": range(len(episode_rewards)),
            "reward": episode_rewards,
            "action": episode_actions,
            "risk_multiplier": episode_risk_multipliers,
            "episode": episode,
        })
        all_results.append(episode_df)

    results_df = pd.concat(all_results, ignore_index=True)
    metrics = {
        "total_episodes": episodes,
        "avg_reward": results_df["reward"].mean(),
        "std_reward": results_df["reward"].std(),
        "total_reward": results_df["reward"].sum(),
        "avg_risk_multiplier": results_df["risk_multiplier"].mean(),
    }

    if verbose:
        print("\nBacktest Results:")
        print(f"Average Reward: {metrics['avg_reward']:.4f}")
        print(f"Total Reward: {metrics['total_reward']:.4f}")
        print(f"Average Risk Multiplier: {metrics['avg_risk_multiplier']:.4f}")

    return results_df, metrics


def run_rl_overlay_backtest(
    df: pd.DataFrame,
    agent: DQNAgent,
    env_class,
    *,
    base_position_size: np.ndarray | None = None,
    reference_risk: float = 0.01,
    commission: float = 0.0006,
    slippage: float = 0.0002,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Один проход по данным: множитель риска из политики масштабирует ``position_size``
    в ``Backtester`` (согласованно с ``ExecutionManager``: base size × RL множитель).
    """
    from backtesting.backtester import Backtester
    from backtesting.performance_metrics import PerformanceMetrics

    n = len(df)
    if base_position_size is None:
        base_position_size = np.ones(n, dtype=float)
    base_position_size = np.asarray(base_position_size, dtype=float).reshape(-1)
    if base_position_size.shape[0] != n:
        raise ValueError("base_position_size length must match df")

    rl_scale = np.ones(n, dtype=float)
    env = env_class(df.reset_index(drop=True))
    state = env.reset()
    done = False
    while not done:
        i = env.current_step
        action = agent.act(state, training=False)
        m = agent.get_risk_multiplier(action)
        next_state, _, done, _ = env.step(action)
        if i + 1 < n:
            rl_scale[i + 1] = m / reference_risk
        state = next_state

    bt = Backtester(commission=commission, slippage=slippage)
    eff_pos = base_position_size * rl_scale
    perf = bt.run(df[["close"]], df["final_signal"], position_size=eff_pos)
    metrics = PerformanceMetrics(perf).calculate_metrics().to_dict()
    return perf, metrics


def run_static_backtest(
    df: pd.DataFrame,
    static_risk: float = 0.01,
    commission: float = 0.0006,
    slippage: float = 0.0002,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Базовый бэктест с фиксированным масштабом позиции через ``Backtester`` (не lookahead).
    """
    from backtesting.backtester import Backtester
    from backtesting.performance_metrics import PerformanceMetrics

    n = len(df)
    bt = Backtester(commission=commission, slippage=slippage)
    perf = bt.run(
        df[["close"]],
        df["final_signal"],
        position_size=np.full(n, static_risk),
    )
    metrics = PerformanceMetrics(perf).calculate_metrics().to_dict()
    return perf, metrics


def compare_rl_vs_static(
    df: pd.DataFrame,
    agent: DQNAgent,
    env_class,
    static_risk: float = 0.01,
    episodes: int = 5,
) -> Dict[str, Any]:
    """
    Сравнение: ``run_rl_overlay_backtest`` vs статический множитель (через общий Backtester).

    ``episodes`` — дополнительно прогоняется ``run_rl_backtest`` (награды среды) для диагностики обучения.
    """
    rl_perf, rl_metrics = run_rl_overlay_backtest(df, agent, env_class)
    static_perf, static_metrics = run_static_backtest(df, static_risk=static_risk)
    _, episode_metrics = run_rl_backtest(df, agent, env_class, episodes=episodes, verbose=False)

    rl_tr = rl_metrics.get("Total Return (%)", 0)
    st_tr = static_metrics.get("Total Return (%)", 0)
    return {
        "rl": rl_metrics,
        "static": static_metrics,
        "rl_equity_curve": rl_perf["cum_strategy_returns"],
        "static_equity_curve": static_perf["cum_strategy_returns"],
        "rl_episode_metrics": episode_metrics,
        "improvement": {
            "total_return_pct": rl_tr - st_tr,
            "improvement_pct": (rl_tr - st_tr) / abs(st_tr) * 100 if st_tr != 0 else 0.0,
        },
    }


def train_agent(
    df: pd.DataFrame,
    env_class,
    episodes: int = 5,
    steps_per_episode: int = 500,
    target_update_freq: int = 10,
    verbose: bool = True,
) -> DQNAgent:
    env = env_class(df)
    agent = DQNAgent(state_size=env.state_size, action_size=env.action_size)
    episode_rewards = []

    for episode in range(episodes):
        if verbose:
            print(f"Training Episode {episode + 1}/{episodes}")

        state = env.reset()
        total_reward = 0
        step = 0

        while step < steps_per_episode:
            action = agent.act(state, training=True)
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            total_reward += reward
            state = next_state
            step += 1
            if done:
                break

        if episode % target_update_freq == 0:
            agent.update_target_model()

        episode_rewards.append(total_reward)

        if verbose:
            print(f"Episode {episode + 1} - Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}")

    if verbose:
        print(f"\nTraining completed. Average reward: {np.mean(episode_rewards):.2f}")

    return agent


def plot_comparison(rl_results: pd.DataFrame, static_results: pd.DataFrame, save_path: str = None):
    """График сравнения (matplotlib подключается лениво)."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    axes[0].plot(rl_results["reward"].cumsum(), label="RL Dynamic Risk", alpha=0.7)
    if "net_returns" in static_results.columns:
        axes[0].plot(static_results["net_returns"].cumsum(), label="Static (net_returns)", alpha=0.7)
    axes[0].set_title("Cumulative returns (RL env rewards vs static net_returns)")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Cumulative")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if "risk_multiplier" in rl_results.columns:
        axes[1].hist(rl_results["risk_multiplier"], bins=30, alpha=0.7, label="RL Risk Multiplier")
        axes[1].axvline(0.01, color="red", linestyle="--", label="Ref 1%")
        axes[1].set_title("Risk Multiplier Distribution")
        axes[1].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
