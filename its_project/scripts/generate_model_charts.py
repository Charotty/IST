#!/usr/bin/env python3
"""
Generate Example Model Accuracy Charts for Presentation
======================================================
Creates visualization charts for all neural network models in the project.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
import pandas as pd

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if hasattr(plt.style, 'seaborn-v0_8-whitegrid') else 'ggplot')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Model names in the project
MODELS = ['GRU', 'LSTM', 'Transformer', 'CNN-LOB', 'Siamese-LOB', 'Ensemble', 'Boosting']

# Metrics to visualize
METRICS = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Sharpe Ratio', 'Max Drawdown']


def generate_example_metrics() -> Dict[str, Dict[str, float]]:
    """Generate example metrics for all models."""
    np.random.seed(42)
    
    metrics_data = {}
    
    # Generate realistic metrics for each model
    for model in MODELS:
        if model == 'Ensemble':
            # Ensemble typically performs best
            metrics_data[model] = {
                'Accuracy': 0.72 + np.random.uniform(0.02, 0.05),
                'Precision': 0.70 + np.random.uniform(0.02, 0.05),
                'Recall': 0.68 + np.random.uniform(0.02, 0.05),
                'F1-Score': 0.69 + np.random.uniform(0.02, 0.05),
                'Sharpe Ratio': 1.8 + np.random.uniform(0.2, 0.5),
                'Max Drawdown': -0.15 + np.random.uniform(-0.05, -0.02)
            }
        elif model == 'Transformer':
            # Transformer performs well
            metrics_data[model] = {
                'Accuracy': 0.68 + np.random.uniform(0.02, 0.04),
                'Precision': 0.66 + np.random.uniform(0.02, 0.04),
                'Recall': 0.65 + np.random.uniform(0.02, 0.04),
                'F1-Score': 0.65 + np.random.uniform(0.02, 0.04),
                'Sharpe Ratio': 1.5 + np.random.uniform(0.2, 0.4),
                'Max Drawdown': -0.18 + np.random.uniform(-0.05, -0.02)
            }
        elif model == 'CNN-LOB':
            # CNN-LOB good for LOB data
            metrics_data[model] = {
                'Accuracy': 0.65 + np.random.uniform(0.02, 0.04),
                'Precision': 0.63 + np.random.uniform(0.02, 0.04),
                'Recall': 0.62 + np.random.uniform(0.02, 0.04),
                'F1-Score': 0.62 + np.random.uniform(0.02, 0.04),
                'Sharpe Ratio': 1.3 + np.random.uniform(0.2, 0.4),
                'Max Drawdown': -0.20 + np.random.uniform(-0.05, -0.02)
            }
        elif model == 'GRU':
            # GRU moderate performance
            metrics_data[model] = {
                'Accuracy': 0.62 + np.random.uniform(0.02, 0.04),
                'Precision': 0.60 + np.random.uniform(0.02, 0.04),
                'Recall': 0.59 + np.random.uniform(0.02, 0.04),
                'F1-Score': 0.59 + np.random.uniform(0.02, 0.04),
                'Sharpe Ratio': 1.1 + np.random.uniform(0.2, 0.4),
                'Max Drawdown': -0.22 + np.random.uniform(-0.05, -0.02)
            }
        elif model == 'LSTM':
            # LSTM similar to GRU
            metrics_data[model] = {
                'Accuracy': 0.61 + np.random.uniform(0.02, 0.04),
                'Precision': 0.59 + np.random.uniform(0.02, 0.04),
                'Recall': 0.58 + np.random.uniform(0.02, 0.04),
                'F1-Score': 0.58 + np.random.uniform(0.02, 0.04),
                'Sharpe Ratio': 1.0 + np.random.uniform(0.2, 0.4),
                'Max Drawdown': -0.23 + np.random.uniform(-0.05, -0.02)
            }
        elif model == 'Siamese-LOB':
            # Siamese for pattern matching
            metrics_data[model] = {
                'Accuracy': 0.60 + np.random.uniform(0.02, 0.04),
                'Precision': 0.58 + np.random.uniform(0.02, 0.04),
                'Recall': 0.57 + np.random.uniform(0.02, 0.04),
                'F1-Score': 0.57 + np.random.uniform(0.02, 0.04),
                'Sharpe Ratio': 0.9 + np.random.uniform(0.2, 0.4),
                'Max Drawdown': -0.25 + np.random.uniform(-0.05, -0.02)
            }
        elif model == 'Boosting':
            # Boosting (XGBoost/LightGBM)
            metrics_data[model] = {
                'Accuracy': 0.67 + np.random.uniform(0.02, 0.04),
                'Precision': 0.65 + np.random.uniform(0.02, 0.04),
                'Recall': 0.64 + np.random.uniform(0.02, 0.04),
                'F1-Score': 0.64 + np.random.uniform(0.02, 0.04),
                'Sharpe Ratio': 1.4 + np.random.uniform(0.2, 0.4),
                'Max Drawdown': -0.19 + np.random.uniform(-0.05, -0.02)
            }
    
    return metrics_data


def plot_metrics_comparison(metrics_data: Dict[str, Dict[str, float]], output_dir: Path):
    """Create bar chart comparing all metrics across models."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')

    # Flatten axes
    axes = axes.flatten()

    # Color palette - use standard colormap
    colors = plt.cm.tab10(np.linspace(0, 1, len(METRICS)))

    # Plot each metric
    for idx, metric in enumerate(METRICS):
        ax = axes[idx]

        values = [metrics_data[model][metric] for model in MODELS]

        bars = ax.bar(MODELS, values, color=colors[idx], alpha=0.8)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)

        ax.set_title(metric, fontweight='bold')
        ax.set_ylabel('Value')
        ax.grid(axis='y', alpha=0.3)
        ax.set_xticklabels(MODELS, rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: metrics_comparison.png")


def plot_training_curves(output_dir: Path):
    """Create training curves (loss and accuracy over epochs)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Training Curves - Model Learning Progress', fontsize=16, fontweight='bold')
    
    epochs = np.arange(1, 51)
    
    # Loss curves
    ax1 = axes[0]
    for model in MODELS:
        np.random.seed(hash(model) % 2**32)
        
        # Different starting points and convergence rates
        if model == 'Ensemble':
            start_loss = 1.2
            decay = 0.02
        elif model == 'Transformer':
            start_loss = 1.4
            decay = 0.018
        elif model == 'CNN-LOB':
            start_loss = 1.5
            decay = 0.015
        else:
            start_loss = 1.6 + np.random.uniform(0, 0.2)
            decay = 0.012 + np.random.uniform(0, 0.005)
        
        loss = start_loss * np.exp(-decay * epochs) + 0.1 + np.random.normal(0, 0.02, len(epochs))
        ax1.plot(epochs, loss, label=model, linewidth=2, alpha=0.8)
    
    ax1.set_xlabel('Epoch', fontweight='bold')
    ax1.set_ylabel('Loss', fontweight='bold')
    ax1.set_title('Training Loss', fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(alpha=0.3)
    
    # Accuracy curves
    ax2 = axes[1]
    for model in MODELS:
        np.random.seed(hash(model) % 2**32 + 1)
        
        # Different starting points and convergence rates
        if model == 'Ensemble':
            start_acc = 0.45
            growth = 0.008
        elif model == 'Transformer':
            start_acc = 0.42
            growth = 0.007
        elif model == 'CNN-LOB':
            start_acc = 0.40
            growth = 0.006
        else:
            start_acc = 0.38 + np.random.uniform(0, 0.05)
            growth = 0.005 + np.random.uniform(0, 0.002)
        
        acc = start_acc + growth * epochs - 0.0001 * epochs**2 + np.random.normal(0, 0.01, len(epochs))
        acc = np.clip(acc, 0, 1)
        ax2.plot(epochs, acc, label=model, linewidth=2, alpha=0.8)
    
    ax2.set_xlabel('Epoch', fontweight='bold')
    ax2.set_ylabel('Accuracy', fontweight='bold')
    ax2.set_title('Training Accuracy', fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: training_curves.png")


def plot_radar_chart(metrics_data: Dict[str, Dict[str, float]], output_dir: Path):
    """Create radar chart for multi-metric comparison."""
    from math import pi

    # Select metrics for radar (exclude Max Drawdown as it's negative)
    radar_metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Sharpe Ratio']

    # Normalize Sharpe Ratio to 0-1 range (assuming max 3.0)
    for model in metrics_data:
        metrics_data[model]['Sharpe Ratio Normalized'] = min(metrics_data[model]['Sharpe Ratio'] / 3.0, 1.0)

    radar_metrics_normalized = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Sharpe Ratio Normalized']

    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))

    # Number of variables
    N = len(radar_metrics_normalized)

    # Angles for each axis
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Plot each model
    colors = plt.cm.tab10(np.linspace(0, 1, len(MODELS)))

    for idx, model in enumerate(MODELS):
        values = [metrics_data[model][m] for m in radar_metrics_normalized]
        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[idx], alpha=0.8)
        ax.fill(angles, values, alpha=0.15, color=colors[idx])

    # Add labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Sharpe Ratio'], fontweight='bold')

    # Set y-axis limits
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
    ax.grid(True)

    plt.title('Model Performance Radar Chart', fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(output_dir / 'radar_chart.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: radar_chart.png")


def plot_cumulative_returns(output_dir: Path):
    """Create cumulative returns chart for backtesting."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    days = np.arange(1, 101)
    
    for model in MODELS:
        np.random.seed(hash(model) % 2**32 + 2)
        
        # Different return profiles
        if model == 'Ensemble':
            daily_return = 0.0015
            volatility = 0.02
        elif model == 'Transformer':
            daily_return = 0.0012
            volatility = 0.022
        elif model == 'CNN-LOB':
            daily_return = 0.0010
            volatility = 0.025
        elif model == 'Boosting':
            daily_return = 0.0011
            volatility = 0.023
        else:
            daily_return = 0.0005 + np.random.uniform(0, 0.0005)
            volatility = 0.028 + np.random.uniform(0, 0.005)
        
        returns = np.random.normal(daily_return, volatility, len(days))
        cumulative = np.cumprod(1 + returns) - 1
        
        ax.plot(days, cumulative, label=model, linewidth=2, alpha=0.8)
    
    # Add benchmark (BTC buy & hold)
    np.random.seed(999)
    btc_returns = np.random.normal(0.0008, 0.035, len(days))
    btc_cumulative = np.cumprod(1 + btc_returns) - 1
    ax.plot(days, btc_cumulative, label='BTC Buy & Hold', linewidth=2, alpha=0.8, linestyle='--', color='black')
    
    ax.set_xlabel('Days', fontweight='bold')
    ax.set_ylabel('Cumulative Return', fontweight='bold')
    ax.set_title('Cumulative Returns - 100 Day Backtest', fontsize=16, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cumulative_returns.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: cumulative_returns.png")


def plot_confusion_matrices(output_dir: Path):
    """Create confusion matrices for each model."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Confusion Matrices - Classification Performance', fontsize=16, fontweight='bold')

    axes = axes.flatten()

    classes = ['SELL', 'HOLD', 'BUY']

    for idx, model in enumerate(MODELS):
        if idx >= len(axes):
            break

        np.random.seed(hash(model) % 2**32 + 3)

        # Generate confusion matrix
        if model == 'Ensemble':
            base_accuracy = 0.72
        elif model == 'Transformer':
            base_accuracy = 0.68
        elif model == 'CNN-LOB':
            base_accuracy = 0.65
        else:
            base_accuracy = 0.60 + np.random.uniform(0, 0.05)

        # Create confusion matrix
        cm = np.random.dirichlet([1, 1, 1], size=3) * 100 * base_accuracy
        cm = cm + np.random.uniform(0, 5, (3, 3))
        cm = cm / cm.sum(axis=1, keepdims=True) * 100

        # Plot heatmap with matplotlib
        im = axes[idx].imshow(cm, cmap='Blues', aspect='auto')
        axes[idx].set_xticks(np.arange(len(classes)))
        axes[idx].set_yticks(np.arange(len(classes)))
        axes[idx].set_xticklabels(classes)
        axes[idx].set_yticklabels(classes)

        # Add text annotations
        for i in range(len(classes)):
            for j in range(len(classes)):
                text = axes[idx].text(j, i, f'{cm[i, j]:.1f}',
                                   ha="center", va="center", color="black", fontsize=10)

        axes[idx].set_title(model, fontweight='bold')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')

    # Remove empty subplot
    if len(MODELS) < len(axes):
        fig.delaxes(axes[-1])

    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: confusion_matrices.png")


def plot_feature_importance(output_dir: Path):
    """Create feature importance chart."""
    fig, ax = plt.subplots(figsize=(14, 8))

    # Example features
    features = [
        'Orderbook Imbalance',
        'Spread',
        'Mid Price',
        'VWAP',
        'Order Flow',
        'Price Impact',
        'SMA',
        'EMA',
        'RSI',
        'MACD',
        'Bollinger Bands',
        'ATR',
        'Trade Intensity',
        'Volatility',
        'LOB Depth'
    ]

    # Generate importance values
    np.random.seed(42)
    importance = np.random.dirichlet(np.ones(len(features)), size=1)[0] * 100
    importance = np.sort(importance)[::-1]

    # Sort features by importance
    sorted_indices = np.argsort(importance)[::-1]
    sorted_features = [features[i] for i in sorted_indices]
    sorted_importance = importance[sorted_indices]

    # Plot horizontal bar chart
    colors = plt.cm.viridis(np.linspace(0, 1, len(features)))
    bars = ax.barh(sorted_features, sorted_importance, color=colors)

    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.5, bar.get_y() + bar.get_height()/2,
               f'{width:.1f}%',
               ha='left', va='center', fontsize=9)

    ax.set_xlabel('Importance (%)', fontweight='bold')
    ax.set_ylabel('Feature', fontweight='bold')
    ax.set_title('Feature Importance - Global SHAP Values', fontsize=16, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim(0, max(sorted_importance) * 1.15)

    plt.tight_layout()
    plt.savefig(output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: feature_importance.png")


def main():
    """Generate all charts."""
    # Create output directory
    output_dir = Path(__file__).parent.parent / 'presentation_charts'
    output_dir.mkdir(exist_ok=True)
    
    print(f"📊 Generating model performance charts...")
    print(f"📁 Output directory: {output_dir}")
    print()
    
    # Generate metrics data
    metrics_data = generate_example_metrics()
    
    # Create charts
    plot_metrics_comparison(metrics_data, output_dir)
    plot_training_curves(output_dir)
    plot_radar_chart(metrics_data, output_dir)
    plot_cumulative_returns(output_dir)
    plot_confusion_matrices(output_dir)
    plot_feature_importance(output_dir)
    
    # Create summary table
    df = pd.DataFrame(metrics_data).T
    df = df[METRICS]
    df.to_csv(output_dir / 'metrics_summary.csv', float_format='%.4f')
    print(f"✅ Saved: metrics_summary.csv")
    
    print()
    print(f"✅ All charts generated successfully!")
    print(f"📁 Location: {output_dir}")
    print()
    print("Generated files:")
    print("  - metrics_comparison.png")
    print("  - training_curves.png")
    print("  - radar_chart.png")
    print("  - cumulative_returns.png")
    print("  - confusion_matrices.png")
    print("  - feature_importance.png")
    print("  - metrics_summary.csv")


if __name__ == "__main__":
    main()
