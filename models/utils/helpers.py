"""
Utilities Module

General utilities for model operations.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
import pickle
import json


def save_model(model, filepath: str):
    """
    Save a model to file.
    
    :param model: Model instance
    :param filepath: Path to save file
    """
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)


def load_model(filepath: str):
    """
    Load a model from file.
    
    :param filepath: Path to model file
    :return: Loaded model
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def calculate_feature_importance(model, feature_names: List[str]) -> Dict[str, float]:
    """
    Calculate and return feature importance.
    
    :param model: Model with feature_importances_ attribute
    :param feature_names: List of feature names
    :return: Dict of feature importances
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        return dict(zip(feature_names, importances))
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
        return dict(zip(feature_names, importances))
    else:
        return {}


def normalize_features(df: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
    """
    Normalize features.
    
    :param df: DataFrame with features
    :param method: Normalization method ('standard', 'minmax')
    :return: Normalized DataFrame
    """
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    
    df_normalized = df.copy()
    
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    numeric_cols = df_normalized.select_dtypes(include=[np.number]).columns
    df_normalized[numeric_cols] = scaler.fit_transform(df_normalized[numeric_cols])
    
    return df_normalized


def calculate_rolling_metrics(series: pd.Series, window: int = 20) -> Dict[str, float]:
    """
    Calculate rolling metrics for a series.
    
    :param series: Pandas series
    :param window: Rolling window size
    :return: Dict with rolling metrics
    """
    return {
        'mean': series.rolling(window).mean().iloc[-1],
        'std': series.rolling(window).std().iloc[-1],
        'min': series.rolling(window).min().iloc[-1],
        'max': series.rolling(window).max().iloc[-1],
        'current': series.iloc[-1]
    }


def detect_drift(reference_data: pd.Series, current_data: pd.Series, 
                 threshold: float = 0.05) -> Dict[str, Any]:
    """
    Detect concept drift between reference and current data.
    
    :param reference_data: Reference distribution
    :param current_data: Current distribution
    :param threshold: Drift threshold
    :return: Dict with drift information
    """
    from scipy import stats
    
    # Kolmogorov-Smirnov test
    ks_statistic, ks_pvalue = stats.ks_2samp(reference_data, current_data)
    
    # Mann-Whitney U test
    mw_statistic, mw_pvalue = stats.mannwhitneyu(reference_data, current_data)
    
    drift_detected = ks_pvalue < threshold
    
    return {
        'drift_detected': drift_detected,
        'ks_statistic': ks_statistic,
        'ks_pvalue': ks_pvalue,
        'mw_statistic': mw_statistic,
        'mw_pvalue': mw_pvalue
    }


def create_feature_combinations(features: List[str], max_combinations: int = 3) -> List[List[str]]:
    """
    Create feature combinations for interaction terms.
    
    :param features: List of feature names
    :param max_combinations: Maximum number of features per combination
    :return: List of feature combinations
    """
    from itertools import combinations
    
    all_combinations = []
    
    for i in range(2, min(max_combinations + 1, len(features) + 1)):
        all_combinations.extend(list(combinations(features, i)))
    
    return [list(combo) for combo in all_combinations]


def validate_data(df: pd.DataFrame, required_columns: List[str]) -> Dict[str, Any]:
    """
    Validate DataFrame has required columns and no missing values.
    
    :param df: DataFrame to validate
    :param required_columns: List of required column names
    :return: Dict with validation results
    """
    validation_results = {
        'is_valid': True,
        'missing_columns': [],
        'columns_with_nulls': [],
        'null_counts': {}
    }
    
    # Check for missing columns
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        validation_results['is_valid'] = False
        validation_results['missing_columns'] = list(missing_cols)
    
    # Check for null values
    null_counts = df.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0].index.tolist()
    
    if cols_with_nulls:
        validation_results['columns_with_nulls'] = cols_with_nulls
        validation_results['null_counts'] = null_counts[cols_with_nulls].to_dict()
    
    return validation_results
