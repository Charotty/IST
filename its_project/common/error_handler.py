from __future__ import annotations

import asyncio
import logging
import traceback
from typing import Callable, Type, Any, Optional, Tuple, Union
from functools import wraps
from dataclasses import dataclass, field
from datetime import datetime
import sys

logger = logging.getLogger(__name__)


@dataclass
class ErrorContext:
    """Context for error tracking."""
    error_type: str
    error_message: str
    timestamp: datetime
    traceback_str: str
    component: str
    additional_info: dict = field(default_factory=dict)


@dataclass
class RetryConfig:
    """Configuration for retry logic."""
    max_attempts: int = 3
    base_delay_ms: int = 1000
    max_delay_ms: int = 30000
    exponential_base: float = 2.0
    jitter: bool = True


class GlobalExceptionHandler:
    """
    Centralized error handling with retry logic and global exception handler.
    """

    def __init__(self) -> None:
        self._error_history: list[ErrorContext] = []
        self._error_callbacks: list[Callable[[ErrorContext], None]] = []
        self._setup_global_handler()

    def _setup_global_handler(self) -> None:
        """Setup global exception handler for unhandled exceptions."""
        def handle_exception(exc_type, exc_value, exc_traceback):
            if issubclass(exc_type, KeyboardInterrupt):
                sys.__excepthook__(exc_type, exc_value, exc_traceback)
                return

            error_context = ErrorContext(
                error_type=exc_type.__name__,
                error_message=str(exc_value),
                timestamp=datetime.now(),
                traceback_str="".join(traceback.format_exception(exc_type, exc_value, exc_traceback)),
                component="global",
            )
            self._log_error(error_context)
            self._notify_callbacks(error_context)

        sys.excepthook = handle_exception

    def register_callback(self, callback: Callable[[ErrorContext], None]) -> None:
        """Register callback for error notifications."""
        self._error_callbacks.append(callback)

    def _log_error(self, context: ErrorContext) -> None:
        """Log error with context."""
        logger.error(
            f"Error in {context.component}: {context.error_type} - {context.error_message}\n"
            f"Traceback: {context.traceback_str}"
        )

    def _notify_callbacks(self, context: ErrorContext) -> None:
        """Notify registered callbacks."""
        for callback in self._error_callbacks:
            try:
                callback(context)
            except Exception as e:
                logger.error(f"Error in error callback: {e}")

    def record_error(
        self,
        error: Exception,
        component: str,
        additional_info: Optional[dict] = None,
    ) -> ErrorContext:
        """Record an error with context."""
        context = ErrorContext(
            error_type=type(error).__name__,
            error_message=str(error),
            timestamp=datetime.now(),
            traceback_str=traceback.format_exc(),
            component=component,
            additional_info=additional_info or {},
        )
        self._error_history.append(context)
        self._log_error(context)
        self._notify_callbacks(context)
        return context

    def get_error_history(self, limit: Optional[int] = None) -> list[ErrorContext]:
        """Get error history."""
        if limit:
            return self._error_history[-limit:]
        return self._error_history.copy()

    def clear_history(self) -> None:
        """Clear error history."""
        self._error_history.clear()


def calculate_delay(attempt: int, config: RetryConfig) -> float:
    """Calculate delay with exponential backoff and optional jitter."""
    delay = min(
        config.base_delay_ms * (config.exponential_base ** (attempt - 1)),
        config.max_delay_ms,
    )
    
    if config.jitter:
        import random
        delay = delay * (0.5 + random.random() * 0.5)
    
    return delay / 1000.0  # Convert to seconds


def retry_with_backoff(
    config: Optional[RetryConfig] = None,
    exception_types: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception,
    on_retry: Optional[Callable[[int, Exception], None]] = None,
):
    """
    Decorator for retry logic with exponential backoff.
    
    Args:
        config: Retry configuration
        exception_types: Exception types to catch and retry
        on_retry: Callback called on each retry attempt
    """
    if config is None:
        config = RetryConfig()

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            last_exception = None
            for attempt in range(1, config.max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exception_types as e:
                    last_exception = e
                    if attempt == config.max_attempts:
                        logger.error(
                            f"Function {func.__name__} failed after {config.max_attempts} attempts: {e}"
                        )
                        raise
                    
                    delay = calculate_delay(attempt, config)
                    logger.warning(
                        f"Attempt {attempt}/{config.max_attempts} for {func.__name__} failed: {e}. "
                        f"Retrying in {delay:.2f}s"
                    )
                    
                    if on_retry:
                        on_retry(attempt, e)
                    
                    await asyncio.sleep(delay)
            
            raise last_exception  # type: ignore

        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            import time
            last_exception = None
            for attempt in range(1, config.max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exception_types as e:
                    last_exception = e
                    if attempt == config.max_attempts:
                        logger.error(
                            f"Function {func.__name__} failed after {config.max_attempts} attempts: {e}"
                        )
                        raise
                    
                    delay = calculate_delay(attempt, config)
                    logger.warning(
                        f"Attempt {attempt}/{config.max_attempts} for {func.__name__} failed: {e}. "
                        f"Retrying in {delay:.2f}s"
                    )
                    
                    if on_retry:
                        on_retry(attempt, e)
                    
                    time.sleep(delay)
            
            raise last_exception  # type: ignore

        # Return appropriate wrapper based on whether function is async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def handle_errors(
    component: str,
    default_return: Any = None,
    log_level: int = logging.ERROR,
):
    """
    Decorator for centralized error handling.
    
    Args:
        component: Component name for error tracking
        default_return: Value to return on error
        log_level: Logging level for errors
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.log(
                    log_level,
                    f"Error in {component}.{func.__name__}: {e}",
                    exc_info=True,
                )
                return default_return

        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.log(
                    log_level,
                    f"Error in {component}.{func.__name__}: {e}",
                    exc_info=True,
                )
                return default_return

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


# Global instance
global_error_handler = GlobalExceptionHandler()
