"""
Logger Configuration

Унифицированная конфигурация логирования для всей системы.
"""

import logging
import logging.config
import sys
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime


class LoggerConfig:
    """Конфигурация логирования"""
    
    @staticmethod
    def default_config() -> Dict[str, Any]:
        """Конфигурация по умолчанию"""
        return {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'standard': {
                    'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
                },
                'detailed': {
                    'format': '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s'
                },
                'json': {
                    'format': '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}'
                }
            },
            'handlers': {
                'console': {
                    'class': 'logging.StreamHandler',
                    'level': 'INFO',
                    'formatter': 'standard',
                    'stream': 'ext://sys.stdout'
                },
                'file': {
                    'class': 'logging.handlers.RotatingFileHandler',
                    'level': 'DEBUG',
                    'formatter': 'detailed',
                    'filename': 'trading_system.log',
                    'maxBytes': 10485760,  # 10MB
                    'backupCount': 5,
                    'encoding': 'utf-8'
                },
                'error_file': {
                    'class': 'logging.handlers.RotatingFileHandler',
                    'level': 'ERROR',
                    'formatter': 'json',
                    'filename': 'trading_system_errors.log',
                    'maxBytes': 10485760,  # 10MB
                    'backupCount': 3,
                    'encoding': 'utf-8'
                }
            },
            'loggers': {
                '': {  # root logger
                    'handlers': ['console', 'file'],
                    'level': 'DEBUG',
                    'propagate': False
                },
                'data_layer': {
                    'handlers': ['console', 'file'],
                    'level': 'DEBUG',
                    'propagate': False
                },
                'models': {
                    'handlers': ['console', 'file'],
                    'level': 'INFO',
                    'propagate': False
                },
                'trading': {
                    'handlers': ['console', 'file', 'error_file'],
                    'level': 'DEBUG',
                    'propagate': False
                }
            }
        }
    
    @staticmethod
    def setup_logging(config: Dict[str, Any]) -> None:
        """
        Настройка логирования на основе конфигурации
        
        Args:
            config: Конфигурация логирования
        """
        try:
            logging.config.dictConfig(config)
        except Exception as e:
            # Fallback to basic logging
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            )
            print(f"Failed to setup logging: {e}")
    
    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        """
        Получение логгера с заданным именем
        
        Args:
            name: Имя логгера
            
        Returns:
            logging.Logger: Настроенный логгер
        """
        return logging.getLogger(name)
    
    @staticmethod
    def create_file_handler(
        filename: str,
        level: str = 'DEBUG',
        format_type: str = 'detailed'
    ) -> logging.Handler:
        """
        Создание файлового обработчика
        
        Args:
            filename: Имя файла
            level: Уровень логирования
            format_type: Тип формата
            
        Returns:
            logging.Handler: Файловый обработчик
        """
        # Создание директории если необходимо
        log_path = Path(filename)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        handler = logging.handlers.RotatingFileHandler(
            filename=filename,
            maxBytes=10485760,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        
        handler.setLevel(getattr(logging, level.upper()))
        
        # Форматер
        if format_type == 'json':
            formatter = logging.Formatter(
                '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}'
            )
        else:
            formatter = logging.Formatter(
                '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s'
            )
        
        handler.setFormatter(formatter)
        return handler
    
    @staticmethod
    def create_console_handler(
        level: str = 'INFO',
        format_type: str = 'standard'
    ) -> logging.Handler:
        """
        Создание консольного обработчика
        
        Args:
            level: Уровень логирования
            format_type: Тип формата
            
        Returns:
            logging.Handler: Консольный обработчик
        """
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(getattr(logging, level.upper()))
        
        # Форматер
        if format_type == 'detailed':
            formatter = logging.Formatter(
                '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s'
            )
        else:
            formatter = logging.Formatter(
                '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            )
        
        handler.setFormatter(formatter)
        return handler
    
    @staticmethod
    def setup_structured_logging(
        log_dir: str = './logs',
        service_name: str = 'trading_system'
    ) -> None:
        """
        Настройка структурированного логирования
        
        Args:
            log_dir: Директория для логов
            service_name: Имя сервиса
        """
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        config = {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'json': {
                    'format': '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}'
                }
            },
            'handlers': {
                'json_file': {
                    'class': 'logging.handlers.RotatingFileHandler',
                    'level': 'INFO',
                    'formatter': 'json',
                    'filename': str(log_path / f'{service_name}.json'),
                    'maxBytes': 50 * 1024 * 1024,  # 50MB
                    'backupCount': 10,
                    'encoding': 'utf-8'
                },
                'error_json_file': {
                    'class': 'logging.handlers.RotatingFileHandler',
                    'level': 'ERROR',
                    'formatter': 'json',
                    'filename': str(log_path / f'{service_name}_errors.json'),
                    'maxBytes': 10 * 1024 * 1024,  # 10MB
                    'backupCount': 5,
                    'encoding': 'utf-8'
                }
            },
            'loggers': {
                service_name: {
                    'handlers': ['json_file', 'error_json_file'],
                    'level': 'INFO',
                    'propagate': False
                }
            }
        }
        
        logging.config.dictConfig(config)
    
    @staticmethod
    def add_context_filter(logger: logging.Logger, context: Dict[str, Any]) -> None:
        """
        Добавление контекстного фильтра к логгеру
        
        Args:
            logger: Логгер
            context: Контекстные данные
        """
        class ContextFilter(logging.Filter):
            def filter(self, record):
                for key, value in context.items():
                    setattr(record, key, value)
                return True
        
        logger.addFilter(ContextFilter())
    
    @staticmethod
    def setup_performance_logging(logger: logging.Logger) -> None:
        """
        Настройка логирования производительности
        
        Args:
            logger: Логгер для логирования производительности
        """
        class PerformanceFilter(logging.Filter):
            def filter(self, record):
                # Добавление метрик производительности
                if hasattr(record, 'duration'):
                    record.duration_ms = record.duration * 1000
                return True
        
        logger.addFilter(PerformanceFilter())


class TradingLoggerAdapter(logging.LoggerAdapter):
    """Адаптер для логирования торговых операций"""
    
    def __init__(self, logger: logging.Logger, extra: Dict[str, Any] = None):
        super().__init__(logger, extra or {})
    
    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        """Обработка сообщения с контекстом"""
        # Добавление временной метки
        if 'timestamp' not in self.extra:
            self.extra['timestamp'] = datetime.utcnow().isoformat()
        
        # Добавление контекстной информации
        if 'operation' in self.extra:
            msg = f"[{self.extra['operation']}] {msg}"
        
        return msg, kwargs
    
    def log_trade(self, trade: Dict[str, Any]) -> None:
        """
        Логирование торговой операции
        
        Args:
            trade: Данные о сделке
        """
        self.info(
            f"Trade executed: {trade.get('symbol')} {trade.get('side')} "
            f"{trade.get('quantity')} @ {trade.get('price')}",
            extra={
                'operation': 'trade',
                'symbol': trade.get('symbol'),
                'side': trade.get('side'),
                'quantity': trade.get('quantity'),
                'price': trade.get('price'),
                'pnl': trade.get('pnl')
            }
        )
    
    def log_order(self, order: Dict[str, Any]) -> None:
        """
        Логирование ордера
        
        Args:
            order: Данные об ордере
        """
        self.info(
            f"Order {order.get('status')}: {order.get('symbol')} "
            f"{order.get('side')} {order.get('quantity')} @ {order.get('price')}",
            extra={
                'operation': 'order',
                'order_id': order.get('order_id'),
                'symbol': order.get('symbol'),
                'side': order.get('side'),
                'quantity': order.get('quantity'),
                'price': order.get('price'),
                'status': order.get('status'),
                'type': order.get('type')
            }
        )
    
    def log_error(self, error: Exception, context: Dict[str, Any] = None) -> None:
        """
        Логирование ошибки с контекстом
        
        Args:
            error: Исключение
            context: Контекст ошибки
        """
        self.error(
            f"Error in {context.get('operation', 'unknown')}: {str(error)}",
            extra={
                'operation': context.get('operation', 'error'),
                'error_type': type(error).__name__,
                'error_message': str(error),
                'context': context
            },
            exc_info=True
        )
