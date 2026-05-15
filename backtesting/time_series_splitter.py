"""
TimeSeriesSplitter - Логика Walk Forward Validation с расширяющимся окном.
"""

import numpy as np
import pandas as pd


class TimeSeriesSplitter:
    def __init__(self, n_splits=5, train_size=0.7, test_size=None):
        """
        Логика Walk Forward Validation.
        
        :param n_splits: Количество разбиений
        :param train_size: Процент данных для обучения в каждом окне
        :param test_size: Фиксированный размер теста (если None, используется остаток)
        """
        self.n_splits = n_splits
        self.train_size = train_size
        self.test_size = test_size

    def split(self, data):
        """
        Генератор индексов для обучения и теста.
        
        :param data: DataFrame или Series с временными данными
        :yield: (train_chunk, test_chunk) - обучающая и тестовая выборки
        """
        n_samples = len(data)
        # Определяем размер одного фолда (шага)
        fold_size = n_samples // self.n_splits

        for i in range(self.n_splits):
            # Определяем границы текущего окна
            start = 0
            # Expanding window logic: обучающая выборка растет с каждым шагом
            train_end = int((i + 1) * fold_size * self.train_size)
            test_end = (i + 1) * fold_size

            if i == self.n_splits - 1:
                test_end = n_samples

            yield data.iloc[start:train_end], data.iloc[train_end:test_end]
