"""
Walk Forward Validation - Функции для валидации стратегии на временных рядах.
"""

import pandas as pd
import numpy as np

from .time_series_splitter import TimeSeriesSplitter
from .backtester import Backtester
from .performance_metrics import PerformanceMetrics


def run_walk_forward_validation(df, n_splits=5):
    """
    Walk-Forward валидация с DirectionModel.
    
    На каждом фолде: DirectionModel → get_calibrated_signals(0.52) → Backtester
    
    :param df: DataFrame с признаками и ценами
    :param n_splits: Количество фолдов для валидации
    :return: DataFrame с метриками по каждому фолду
    """
    from ..models.direction_model import DirectionModel
    
    splitter = TimeSeriesSplitter(n_splits=n_splits, train_size=0.7)
    all_fold_stats = []

    print(f"--- Starting Walk-Forward Optimization ({n_splits} Folds) ---")

    for i, (train_data, test_data) in enumerate(splitter.split(df)):
        # 1. Initialize and Train Model on current fold
        fold_model = DirectionModel(n_estimators=100)
        fold_model.feature_cols = ['rsi', 'macd_hist', 'ema_slope', 'adx', 'rsi_15m', 'ema_slope_15m', 'rsi_4h', 'adx_4h']

        y_train = fold_model.prepare_labels(train_data)
        fold_model.train(train_data, y_train)

        # 2. Backtest on unseen test data
        signals, _ = fold_model.get_calibrated_signals(test_data, threshold=0.52)
        bt = Backtester(commission=0.0006, slippage=0.0002)
        perf = bt.run(test_data, signals)

        # 3. Calculate metrics for this fold
        metrics = PerformanceMetrics(perf).calculate_metrics()
        metrics['Fold'] = i + 1
        all_fold_stats.append(metrics)

        print(f"Fold {i+1} Sharpe: {metrics['Sharpe Ratio']:.2f}")

    return pd.DataFrame(all_fold_stats)


def run_integrated_wfo(df, n_splits=3, window_size=24, features=None):
    """
    Walk-Forward валидация интегрированной системы.
    
    Переобучение LGBM + короткое DL (3 epochs)
    get_dynamic_meta_signal → integrated_signal → Backtester
    
    :param df: DataFrame с признаками и ценами
    :param n_splits: Количество фолдов для валидации
    :param window_size: Размер окна для DL моделей
    :param features: Список признаков для обучения
    :return: DataFrame с метриками по каждому фолду
    """
    from ..models.direction_model import DirectionModel
    from ..models.dl_model_builder import DLModelBuilder
    
    if features is None:
        features = ['rsi', 'macd_hist', 'ema_slope', 'adx', 'rsi_15m', 'ema_slope_15m', 'rsi_4h', 'adx_4h']
    
    splitter = TimeSeriesSplitter(n_splits=n_splits, train_size=0.7)
    wfo_stats = []

    print(f'--- Starting Walk-Forward: Integrated System ({n_splits} Folds) ---')

    for i, (train_data, test_data) in enumerate(splitter.split(df)):
        # 1. Initialize factory models for this fold
        fold_lstm = DLModelBuilder.build_lstm(window_size, len(features))
        fold_cnn = DLModelBuilder.build_cnn(window_size, len(features))
        fold_trans = DLModelBuilder.build_transformer(window_size, len(features))

        fold_lgb = DirectionModel()
        fold_lgb.feature_cols = features

        # 2. Data prep for fold
        # Prepare sequences for DL models
        def create_sequences(X, y, time_steps):
            Xs, ys = [], []
            for i in range(len(X) - time_steps):
                Xs.append(X.iloc[i:(i + time_steps)].values)
                ys.append(y.iloc[i + time_steps])
            return np.array(Xs), np.array(ys)
        
        X_train_fold, y_train_fold = create_sequences(
            train_data[features], 
            (train_data['close'].shift(-12) > train_data['close']).astype(int).fillna(0), 
            window_size
        )

        # 3. Quick Train (Reduced epochs for validation speed)
        fold_lstm.fit(X_train_fold, y_train_fold, epochs=3, batch_size=128, verbose=0)
        fold_lgb.train(train_data, fold_lgb.prepare_labels(train_data))

        # 4. Generate Integrated Signal on Test Data
        test_data_copy = test_data.copy()
        
        # Use soft signal from LGB for logic consistency
        soft_sig, _ = fold_lgb.get_calibrated_signals(test_data_copy, threshold=0.52)
        test_data_copy['direction_soft_signal'] = soft_sig

        # Final Signal Generation (simplified version without meta_mgmt_prob)
        # Using median threshold on direction probability as proxy
        dir_probs = fold_lgb.model.predict_proba(test_data_copy[features])[:, 1]
        test_data_copy['meta_mgmt_prob'] = dir_probs
        
        thresh = test_data_copy['meta_mgmt_prob'].median()
        integrated_sig = np.where(
            (test_data_copy['meta_mgmt_prob'] > thresh) & (test_data_copy['direction_soft_signal'] != 0),
            test_data_copy['direction_soft_signal'],
            0
        )

        # 5. Backtest Fold
        bt = Backtester(commission=0.0006, slippage=0.0002)
        perf = bt.run(test_data_copy, integrated_sig)
        metrics = PerformanceMetrics(perf).calculate_metrics()
        metrics['Fold'] = i + 1
        wfo_stats.append(metrics)

        print(f'Fold {i+1} Result | Sharpe: {metrics["Sharpe Ratio"]:.2f} | PF: {metrics["Profit Factor"]:.2f}')

    return pd.DataFrame(wfo_stats)
