"""
ExperimentLogger - Логирование экспериментов в JSON файл.
"""

import json
import os
from datetime import datetime


class ExperimentLogger:
    def __init__(self, log_file='experiments_log.json'):
        """
        Инициализация логгера экспериментов.
        
        :param log_file: Путь к файлу логов
        """
        self.log_file = log_file

    def log_experiment(self, name, params, metrics):
        """
        Сохранение параметров и результатов эксперимента.
        
        :param name: Название эксперимента (например, 'Phase_2_MTF_Full_History')
        :param params: Словарь с параметрами (таймфрейм, комиссии и т.д.)
        :param metrics: Серия или словарь с метриками (Sharpe, DD и т.д.)
        """
        log_entry = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'experiment_name': name,
            'parameters': params,
            'metrics': metrics.to_dict() if hasattr(metrics, 'to_dict') else metrics
        }

        # Загрузка существующих логов
        logs = []
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r') as f:
                    logs = json.load(f)
            except:
                logs = []

        # Добавление новой записи
        logs.append(log_entry)

        # Сохранение обновленного списка
        with open(self.log_file, 'w') as f:
            json.dump(logs, f, indent=4)

        print(f"Эксперимент '{name}' успешно сохранен в {self.log_file}")
