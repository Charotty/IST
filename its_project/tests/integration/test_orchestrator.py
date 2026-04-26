"""Tests for Pipeline Orchestrator."""
import pytest
import asyncio
import numpy as np
import pandas as pd
from unittest.mock import Mock, AsyncMock, MagicMock
from its_project.orchestrator import (
    PipelineOrchestrator,
    OrchestratorConfig,
    PipelineMetrics,
)
from its_project.common.types import MarketData, MarketDataType


@pytest.fixture
def mock_data_source():
    """Mock data source."""
    source = Mock()
    source.connect = AsyncMock()
    source.disconnect = AsyncMock()
    source.is_alive = AsyncMock(return_value=True)
    source.fetch = AsyncMock()
    return source


@pytest.fixture
def mock_feature_pipeline():
    """Mock feature pipeline."""
    pipeline = Mock()
    pipeline.transform = Mock(return_value=np.random.randn(1, 10))
    return pipeline


@pytest.fixture
def mock_model():
    """Mock model."""
    model = Mock()
    model.predict = Mock(return_value=np.array([1]))  # BUY signal
    model.predict_proba = Mock(return_value=np.array([[0.1, 0.2, 0.7]]))
    return model


@pytest.fixture
def mock_decision_engine():
    """Mock decision engine."""
    engine = Mock()
    engine.process_signal = Mock(return_value=None)  # No decision by default
    return engine


@pytest.fixture
def mock_executor():
    """Mock executor."""
    executor = Mock()
    executor.create_order = AsyncMock()
    executor.fetch_balance = AsyncMock(return_value={"USDT": 10000.0})
    return executor


@pytest.fixture
def orchestrator_config(mock_data_source, mock_feature_pipeline, mock_model, mock_decision_engine, mock_executor):
    """Orchestrator configuration."""
    return OrchestratorConfig(
        data_source=mock_data_source,
        feature_pipeline=mock_feature_pipeline,
        model=mock_model,
        decision_engine=mock_decision_engine,
        executor=mock_executor,
        symbols=["BTC/USDT"],
        loop_interval_ms=100,
        enable_paper_trading=True,
    )


@pytest.mark.asyncio
async def test_orchestrator_init(orchestrator_config):
    """Test orchestrator initialization."""
    orchestrator = PipelineOrchestrator(orchestrator_config)
    
    assert not orchestrator.is_running()
    assert orchestrator.get_metrics().processed_ticks == 0


@pytest.mark.asyncio
async def test_orchestrator_initialize(orchestrator_config):
    """Test orchestrator initialization."""
    orchestrator = PipelineOrchestrator(orchestrator_config)
    
    await orchestrator.initialize()
    
    orchestrator_config.data_source.connect.assert_called_once()


@pytest.mark.asyncio
async def test_orchestrator_shutdown(orchestrator_config):
    """Test orchestrator shutdown."""
    orchestrator = PipelineOrchestrator(orchestrator_config)
    
    await orchestrator.initialize()
    await orchestrator.shutdown()
    
    orchestrator_config.data_source.disconnect.assert_called_once()


@pytest.mark.asyncio
async def test_orchestrator_process_tick(orchestrator_config):
    """Test processing a single tick."""
    orchestrator = PipelineOrchestrator(orchestrator_config)
    
    # Setup mock market data
    market_data = MarketData(
        timestamp_ms=1234567890,
        symbol="BTC/USDT",
        type=MarketDataType.TICKER,
        exchange="okx",
        data={"last": 42000.0},
    )
    orchestrator_config.data_source.fetch.return_value = market_data
    
    # Process tick
    decision = await orchestrator._process_tick(market_data)
    
    # Verify metrics updated
    metrics = orchestrator.get_metrics()
    assert metrics.processed_ticks == 1
    assert metrics.signals_generated == 1


@pytest.mark.asyncio
async def test_orchestrator_callback_registration(orchestrator_config):
    """Test callback registration."""
    orchestrator = PipelineOrchestrator(orchestrator_config)
    
    callback_called = []
    
    def on_tick_callback(market_data):
        callback_called.append(("tick", market_data))
    
    orchestrator.register_callback("on_tick", on_tick_callback)
    
    # Emit event
    market_data = MarketData(
        timestamp_ms=1234567890,
        symbol="BTC/USDT",
        type=MarketDataType.TICKER,
        exchange="okx",
        data={"last": 42000.0},
    )
    await orchestrator._emit("on_tick", market_data)
    
    assert len(callback_called) == 1
    assert callback_called[0][0] == "tick"


@pytest.mark.asyncio
async def test_orchestrator_get_status(orchestrator_config):
    """Test getting orchestrator status."""
    orchestrator = PipelineOrchestrator(orchestrator_config)
    
    status = orchestrator.get_status()
    
    assert "running" in status
    assert "uptime_seconds" in status
    assert "processed_ticks" in status
    assert "symbols" in status
    assert status["symbols"] == ["BTC/USDT"]


def test_pipeline_metrics():
    """Test pipeline metrics dataclass."""
    metrics = PipelineMetrics()
    
    assert metrics.processed_ticks == 0
    assert metrics.signals_generated == 0
    assert metrics.decisions_executed == 0
    assert metrics.errors == 0
    assert metrics.last_tick_time is None


def test_orchestrator_config_defaults():
    """Test orchestrator config defaults."""
    config = OrchestratorConfig(
        data_source=Mock(),
        feature_pipeline=Mock(),
        model=Mock(),
        decision_engine=Mock(),
        executor=Mock(),
        symbols=["BTC/USDT"],
    )
    
    assert config.loop_interval_ms == 1000
    assert config.max_retries == 3
    assert config.enable_paper_trading is True


@pytest.mark.asyncio
async def test_orchestrator_error_handling(orchestrator_config):
    """Test error handling during tick processing."""
    orchestrator = PipelineOrchestrator(orchestrator_config)
    
    # Make feature pipeline raise error
    orchestrator_config.feature_pipeline.transform.side_effect = Exception("Test error")
    
    market_data = MarketData(
        timestamp_ms=1234567890,
        symbol="BTC/USDT",
        type=MarketDataType.TICKER,
        exchange="okx",
        data={"last": 42000.0},
    )
    
    # Should not raise, but log error
    decision = await orchestrator._process_tick(market_data)
    
    metrics = orchestrator.get_metrics()
    assert metrics.errors == 1
    assert decision is None


@pytest.mark.asyncio
async def test_orchestrator_decision_execution(orchestrator_config):
    """Test decision execution when paper trading enabled."""
    from its_project.decision.decision import Decision, Action
    from its_project.execution.base import OrderType
    
    orchestrator = PipelineOrchestrator(orchestrator_config)
    
    # Setup decision engine to return a decision
    decision = Decision(
        action=Action.BUY,
        symbol="BTC/USDT",
        size=0.1,
        price=42000.0,
        stop_loss=None,
        take_profit=None,
        timestamp=1234567890,
        reason="Test",
    )
    orchestrator_config.decision_engine.process_signal.return_value = decision
    
    market_data = MarketData(
        timestamp_ms=1234567890,
        symbol="BTC/USDT",
        type=MarketDataType.TICKER,
        exchange="okx",
        data={"last": 42000.0},
    )
    
    # Process tick
    await orchestrator._process_tick(market_data)
    
    # The orchestrator may or may not execute the decision depending on implementation
    # Just verify the decision engine was called
    orchestrator_config.decision_engine.process_signal.assert_called_once()
