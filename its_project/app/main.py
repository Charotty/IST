from __future__ import annotations

import asyncio
import logging
import aiohttp

from its_project.common.config import AppConfig, load_config
from its_project.common.logging import configure_logging
from its_project.data_layer.binance_ws import BinanceWsClient, publish_ws_to_queues
from its_project.data_layer.binance_rest import (
    BinanceFuturesRestClient,
    BinanceRestConfig,
    BinanceSpotRestClient,
)
from its_project.data_layer.glassnode import GlassnodeClient, GlassnodeConfig, build_onchain_marketdata
from its_project.data_layer.polling import polling_task
from its_project.data_layer.sentiment_x import XClient, XSentimentConfig, build_sentiment_marketdata
from its_project.pipeline.queues import create_queues
from its_project.storage import TimescaleStorage, ParquetStorage, StorageReadAPI, batch_writer_task
from its_project.models import ModelRegistry
from its_project.metalearning import (
    ModelSelector,
    HyperparameterOptimizer,
    TimeSeriesSplitter,
    MLMetrics,
    TradingMetrics,
)
from its_project.decision import (
    Action,
    Signal,
    Decision,
    TradingDecisionEngine,
)
from its_project.features import (
    synchronize_marketdata,
    extract_ohlcv_from_synced,
    WindowedFeatures,
    FeaturePipeline,
    handle_missing,
    normalize_features,
    TechnicalFeatures,
    OrderBookFeatures,
    MicrostructureFeatures,
    FeatureScaler,
)

logger = logging.getLogger(__name__)


async def _drain(name: str, q: asyncio.Queue) -> None:
    while True:
        md = await q.get()
        q.task_done()
        logger.debug("%s: %s %s", name, md.type, md.symbol)


async def preprocessing_task(
    *,
    read_api,
    window_size: int,
    stride: int,
    out_queue: asyncio.Queue,
    stop_event: asyncio.Event,
    interval_s: float = 10.0,
) -> None:
    """
    Periodically read recent data from storage, synchronize, create windows, compute features, and publish tensors.
    """
    # Feature pipeline (immutable)
    pipeline = FeaturePipeline([
        TechnicalFeatures({"indicators": ["rsi", "macd", "bbands", "atr", "stoch"]}),
        OrderBookFeatures({"depth_levels": 5, "imbalance_levels": 10}),
        MicrostructureFeatures({"window": 20}),
    ])
    scaler = FeatureScaler({"method": "zscore"})

    while not stop_event.is_set():
        try:
            # Read recent data (e.g., last 5 minutes)
            now = pd.Timestamp.now(tz="UTC")
            start = now - pd.Timedelta(minutes=5)
            price_data = await read_api.read("BTCUSDT", start.to_pydatetime(), now.to_pydatetime(), data_type="TRADE")
            lob_data = await read_api.read("BTCUSDT", start.to_pydatetime(), now.to_pydatetime(), data_type="ORDERBOOK")
            onchain_data = await read_api.read("BTCUSDT", start.to_pydatetime(), now.to_pydatetime(), data_type="ONCHAIN")
            sentiment_data = await read_api.read("BTCUSDT", start.to_pydatetime(), now.to_pydatetime(), data_type="SENTIMENT")

            if not (price_data or lob_data):
                await asyncio.sleep(interval_s)
                continue

            # Synchronize
            synced = synchronize_marketdata(price_data, lob_data, onchain_data, sentiment_data, freq="1s", method="ffill")
            if synced.empty:
                await asyncio.sleep(interval_s)
                continue

            # Build OHLCV
            ohlcv = extract_ohlcv_from_synced(synced)
            if ohlcv.empty:
                await asyncio.sleep(interval_s)
                continue

            # Handle missing
            ohlcv_clean = handle_missing(ohlcv, method="ffill", limit=1)

            # Create windows
            window_gen = WindowedFeatures(window_size=window_size, stride=stride)
            windows = list(window_gen.create_windows(ohlcv_clean))
            if not windows:
                await asyncio.sleep(interval_s)
                continue

            # Compute features for each window
            for i, win in enumerate(windows):
                try:
                    # Merge OHLCV with LOB snapshots (if available)
                    # For simplicity, use OHLCV for all features; OrderBookFeatures will handle missing LOB
                    feats = pipeline.fit_transform(win)
                    # Scale features
                    feats_scaled = scaler.fit_transform(feats)
                    # Publish tensor X_t^{(L)}
                    await out_queue.put({
                        "window_id": i,
                        "timestamp": win.index[-1].isoformat(),
                        "features": feats_scaled.tolist(),
                        "feature_names": pipeline.get_feature_names(),
                        "shape": feats_scaled.shape,
                    })
                except Exception:
                    logger.exception("Feature computation error for window %d", i)
            logger.debug("Published %d feature windows to features queue", len(windows))

        except Exception:
            logger.exception("Preprocessing task error")
        await asyncio.sleep(interval_s)


async def predictor_task(
    *,
    model_name: str,
    model_config: dict,
    model_path: str | None,
    in_queue: asyncio.Queue,
    out_queue: asyncio.Queue,
    metalearning_queue: asyncio.Queue,
    stop_event: asyncio.Event,
) -> None:
    """
    Consume feature tensors, predict with loaded model, publish predictions.
    Dynamically updates model when metalearning publishes better model.
    """
    from pathlib import Path
    import numpy as np

    # Initial model
    current_model = None
    if model_path and Path(model_path).exists():
        current_model = ModelRegistry.load_model(Path(model_path))
        logger.info("Loaded initial model from %s", model_path)
    else:
        current_model = ModelRegistry.get_model(model_name, model_config)
        logger.warning("Initial model not fitted; will wait for metalearning")

    while not stop_event.is_set():
        # Check for metalearning updates (non-blocking)
        try:
            meta_msg = await asyncio.wait_for(metalearning_queue.get(), timeout=0.1)
            metalearning_queue.task_done()
            # Load new best model
            new_model_type = meta_msg.get("model_type")
            new_config = meta_msg.get("model_config", {})
            logger.info("Updating model to %s", new_model_type)
            current_model = ModelRegistry.get_model(new_model_type.lower(), new_config)
        except asyncio.TimeoutError:
            pass
        except Exception:
            logger.exception("Error updating model from metalearning")

        if current_model is None or not current_model.is_fitted:
            await asyncio.sleep(1.0)
            continue

        try:
            feat_msg = await asyncio.wait_for(in_queue.get(), timeout=1.0)
            in_queue.task_done()
        except asyncio.TimeoutError:
            continue
        except Exception:
            logger.exception("Error reading from features queue")
            continue

        try:
            # Extract features
            features = np.array(feat_msg["features"])
            # Reshape for deep models if needed (add seq_len=1)
            if current_model.__class__.__name__.lower() in {"lstmmodel", "transformermodel"} and features.ndim == 2:
                features = features[np.newaxis, :, :]  # (1, seq_len, n_features)

            # Predict
            preds = current_model.predict(features)
            proba = current_model.predict_proba(features)
            confidence = current_model.get_confidence(features)

            # Publish prediction
            await out_queue.put({
                "window_id": feat_msg["window_id"],
                "timestamp": feat_msg["timestamp"],
                "prediction": int(preds[0]),
                "probabilities": proba[0].tolist(),
                "confidence": float(confidence[0]),
                "model_type": current_model.__class__.__name__,
            })
            logger.debug(
                "Prediction: %s (conf=%.3f) for window %s",
                preds[0], confidence[0], feat_msg["window_id"]
            )
        except Exception:
            logger.exception("Prediction error for window %s", feat_msg.get("window_id"))


async def metalearning_task(
    *,
    read_api,
    model_names: list[str],
    param_spaces: dict,
    out_queue: asyncio.Queue,
    stop_event: asyncio.Event,
    interval_s: float = 300.0,  # 5 minutes
) -> None:
    """
    Periodically run model selection and hyperparameter optimization.
    """
    from pathlib import Path
    import numpy as np

    while not stop_event.is_set():
        try:
            # Load recent data for meta-learning
            now = pd.Timestamp.now(tz="UTC")
            start = now - pd.Timedelta(hours=24)  # last 24h
            price_data = await read_api.read("BTCUSDT", start.to_pydatetime(), now.to_pydatetime(), data_type="TRADE")
            lob_data = await read_api.read("BTCUSDT", start.to_pydatetime(), now.to_pydatetime(), data_type="ORDERBOOK")
            onchain_data = await read_api.read("BTCUSDT", start.to_pydatetime(), now.to_pydatetime(), data_type="ONCHAIN")
            sentiment_data = await read_api.read("BTCUSDT", start.to_pydatetime(), now.to_pydatetime(), data_type="SENTIMENT")

            if not (price_data or lob_data):
                await asyncio.sleep(interval_s)
                continue

            # Synchronize and extract features
            synced = synchronize_marketdata(price_data, lob_data, onchain_data, sentiment_data, freq="1s", method="ffill")
            if synced.empty:
                await asyncio.sleep(interval_s)
                continue
            ohlcv = extract_ohlcv_from_synced(synced)
            if ohlcv.empty:
                await asyncio.sleep(interval_s)
                continue

            # Prepare features and targets (simple: next period return as target)
            features = ohlcv.dropna()
            if len(features) < 200:
                await asyncio.sleep(interval_s)
                continue

            # Create target: next 5-min return direction
            returns = features["close"].pct_change(periods=300).shift(-300)
            target = np.where(returns > 0.001, 2, np.where(returns < -0.001, 0, 1))  # buy/sell/hold
            target = target[:-300]
            X = features.iloc[:-300].values
            y = target

            if len(np.unique(y)) < 2:
                await asyncio.sleep(interval_s)
                continue

            # Time series split
            splitter = TimeSeriesSplitter(n_splits=3, gap=60)
            splits = list(splitter.split(X))
            if len(splits) < 2:
                await asyncio.sleep(interval_s)
                continue

            train_idx, test_idx = splits[-1]
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Create candidate models
            candidate_models = []
            for name in model_names:
                cfg = param_spaces.get(name, {})
                if name == "ensemble":
                    model = ModelRegistry.get_model(name, cfg)
                else:
                    # Simple config for deep models
                    cfg = {"input_size": X.shape[1], "epochs": 10, "batch_size": 32}
                    model = ModelRegistry.get_model(name, cfg)
                candidate_models.append(model)

            # Model selection
            selector = ModelSelector(
                models=candidate_models,
                metrics=["accuracy", "sharpe_ratio"],
                weights=[0.4, 0.6],
            )
            results_df = selector.evaluate_all(X_train, y_train, X_test, y_test)
            best_model, best_scores = selector.select_best()

            # Publish best model info
            await out_queue.put({
                "timestamp": now.isoformat(),
                "model_type": best_model.__class__.__name__,
                "best_scores": best_scores,
                "ranking": selector.get_ranking().to_dict("records"),
                "model_config": best_model.config,
            })
            logger.info(
                "Meta-learning: best model %s with scores %s",
                best_model.__class__.__name__,
                best_scores,
            )
        except Exception:
            logger.exception("Meta-learning task error")
        await asyncio.sleep(interval_s)


async def decision_task(
    *,
    in_queue: asyncio.Queue,
    out_queue: asyncio.Queue,
    stop_event: asyncio.Event,
) -> None:
    """
    Consume predictions, make trading decisions via TradingDecisionEngine, publish decisions.
    """
    # Initialize decision engine with default config
    decision_engine = TradingDecisionEngine({
        "decision": {"confidence_threshold": 0.7},
        "risk": {
            "max_position_size": 0.1,
            "max_portfolio_risk": 0.05,
            "max_drawdown": 0.15,
            "max_correlation": 0.7,
        },
        "sizing": {"method": "fixed", "size": 0.01},
        "portfolio": {"max_positions": 5},
    })
    account_balance = 10000.0  # TODO: fetch from broker

    while not stop_event.is_set():
        try:
            pred_msg = await asyncio.wait_for(in_queue.get(), timeout=1.0)
            in_queue.task_done()
        except asyncio.TimeoutError:
            continue
        except Exception:
            logger.exception("Error reading from predictions queue")
            continue

        try:
            # Convert prediction to Signal
            action_map = {0: "sell", 1: "hold", 2: "buy"}
            action = Action(action_map.get(pred_msg["prediction"], "hold"))
            signal = Signal(
                action=action,
                confidence=pred_msg["confidence"],
                timestamp=int(pd.Timestamp.now(tz="UTC").timestamp() * 1000),
                symbol=pred_msg.get("symbol", "BTCUSDT"),
                metadata={"model_type": pred_msg.get("model_type")},
            )
            # Simple market state (TODO: fetch real market data)
            market_state = {"price": 42000.0}
            # Process signal
            decision = decision_engine.process_signal(signal, market_state, account_balance)
            if decision:
                await out_queue.put({
                    "timestamp": decision.timestamp,
                    "symbol": decision.symbol,
                    "action": decision.action.value,
                    "size": decision.size,
                    "price": decision.price,
                    "stop_loss": decision.stop_loss,
                    "take_profit": decision.take_profit,
                    "reason": decision.reason,
                    "model_type": pred_msg.get("model_type"),
                })
                logger.info(
                    "Decision: %s %s @ %s (SL=%s, TP=%s)",
                    decision.action.value,
                    decision.symbol,
                    decision.size,
                    decision.stop_loss,
                    decision.take_profit,
                )
        except Exception:
            logger.exception("Decision processing error")


async def run_app(cfg: AppConfig) -> None:
    queues = create_queues(cfg.queue_maxsize)
    stop_event = asyncio.Event()

    # Storage backends
    warm = TimescaleStorage(dsn=cfg.timescale_dsn)
    await warm.connect()
    cold = ParquetStorage(base_path=cfg.parquet_base_path)
    read_api = StorageReadAPI(warm=warm, cold=cold)

    # Batch writers
    storage_tasks = [
        asyncio.create_task(
            batch_writer_task(
                name="price_warm",
                storage=warm,
                batch_size=500,
                max_interval_s=5.0,
                in_queue=queues.price_raw,
                stop_event=stop_event,
            ),
            name="price_warm_writer",
        ),
        asyncio.create_task(
            batch_writer_task(
                name="lob_warm",
                storage=warm,
                batch_size=500,
                max_interval_s=5.0,
                in_queue=queues.lob_raw,
                stop_event=stop_event,
            ),
            name="lob_warm_writer",
        ),
        asyncio.create_task(
            batch_writer_task(
                name="onchain_cold",
                storage=cold,
                batch_size=200,
                max_interval_s=10.0,
                in_queue=queues.onchain_raw,
                stop_event=stop_event,
            ),
            name="onchain_cold_writer",
        ),
        asyncio.create_task(
            batch_writer_task(
                name="sentiment_cold",
                storage=cold,
                batch_size=200,
                max_interval_s=10.0,
                in_queue=queues.sentiment_raw,
                stop_event=stop_event,
            ),
            name="sentiment_cold_writer",
        ),
    ]

    spot_ws = BinanceWsClient(
        base_url=cfg.binance_ws_base_spot,
        symbols=cfg.symbols_spot,
        stream_names=cfg.binance_spot_streams,
        exchange="binance_spot",
    )

    futures_ws = BinanceWsClient(
        base_url=cfg.binance_ws_base_futures,
        symbols=cfg.symbols_futures,
        stream_names=cfg.binance_futures_streams,
        exchange="binance_futures",
    )

    async with aiohttp.ClientSession() as session:
        spot_rest = BinanceSpotRestClient(
            session=session,
            config=BinanceRestConfig(base_url=cfg.binance_rest_base_spot),
        )
        futures_rest = BinanceFuturesRestClient(
            session=session,
            config=BinanceRestConfig(base_url=cfg.binance_rest_base_futures),
        )

        async def fetch_onchain() -> object:
            if not cfg.glassnode_api_key:
                return None

            client = GlassnodeClient(session=session, config=GlassnodeConfig(api_key=cfg.glassnode_api_key))

            payload = {
                "inflow_outflow": await client.fetch_metric(
                    path="/v1/metrics/transactions/transfers_volume_sum",
                    params={"a": "BTC", "i": "1h"},
                )
            }
            return build_onchain_marketdata(symbol="BTCUSDT", exchange="glassnode", payload=payload)

        async def fetch_sentiment() -> object:
            if not cfg.twitter_bearer_token:
                return None

            client = XClient(session=session, config=XSentimentConfig(bearer_token=cfg.twitter_bearer_token))
            data = await client.recent_search(query="bitcoin", max_results=10)
            payload = {"raw": data}
            return build_sentiment_marketdata(symbol="BTCUSDT", exchange="x", payload=payload)

        tasks = storage_tasks + [
            asyncio.create_task(
                publish_ws_to_queues(
                    ws=spot_ws,
                    rest_client=spot_rest,
                    price_queue=queues.price_raw,
                    lob_queue=queues.lob_raw,
                ),
                name="binance_spot_ws",
            ),
            asyncio.create_task(
                publish_ws_to_queues(
                    ws=futures_ws,
                    rest_client=futures_rest,
                    price_queue=queues.price_raw,
                    lob_queue=queues.lob_raw,
                ),
                name="binance_futures_ws",
            ),
            asyncio.create_task(_drain("price_raw", queues.price_raw), name="drain_price"),
            asyncio.create_task(_drain("lob_raw", queues.lob_raw), name="drain_lob"),
        ]

        tasks.append(
            asyncio.create_task(
                polling_task(
                    name="glassnode",
                    interval_s=cfg.glassnode_poll_interval_s,
                    fetch=fetch_onchain,
                    out_queue=queues.onchain_raw,
                    stop_event=stop_event,
                ),
                name="poll_glassnode",
            )
        )
        tasks.append(
            asyncio.create_task(
                polling_task(
                    name="sentiment_x",
                    interval_s=cfg.sentiment_poll_interval_s,
                    fetch=fetch_sentiment,
                    out_queue=queues.sentiment_raw,
                    stop_event=stop_event,
                ),
                name="poll_sentiment_x",
            )
        )

        # Preprocessing task
        tasks.append(
            asyncio.create_task(
                preprocessing_task(
                    read_api=read_api,
                    window_size=60,
                    stride=10,
                    out_queue=queues.features_ready,
                    stop_event=stop_event,
                    interval_s=10.0,
                ),
                name="preprocessing",
            )
        )
        tasks.append(
            asyncio.create_task(_drain("features_ready", queues.features_ready), name="drain_features")
        )

        # Predictor task
        tasks.append(
            asyncio.create_task(
                predictor_task(
                    model_name="ensemble",
                    model_config={"rf_estimators": 100, "rf_max_depth": 10},
                    model_path=None,  # No pre-trained model yet
                    in_queue=queues.features_ready,
                    out_queue=queues.predictions,
                    metalearning_queue=queues.metalearning_results,
                    stop_event=stop_event,
                ),
                name="predictor",
            )
        )
        tasks.append(
            asyncio.create_task(_drain("predictions", queues.predictions), name="drain_predictions")
        )

        # Meta-learning task
        tasks.append(
            asyncio.create_task(
                metalearning_task(
                    read_api=read_api,
                    model_names=["lstm", "transformer", "ensemble"],
                    param_spaces={
                        "ensemble": {"rf_estimators": 100, "rf_max_depth": 10},
                    },
                    out_queue=queues.metalearning_results,
                    stop_event=stop_event,
                    interval_s=300.0,
                ),
                name="metalearning",
            )
        )
        tasks.append(
            asyncio.create_task(_drain("metalearning_results", queues.metalearning_results), name="drain_metalearning")
        )

        # Decision task
        tasks.append(
            asyncio.create_task(
                decision_task(
                    in_queue=queues.predictions,
                    out_queue=queues.decisions,
                    stop_event=stop_event,
                ),
                name="decision",
            )
        )
        tasks.append(
            asyncio.create_task(_drain("decisions", queues.decisions), name="drain_decisions")
        )

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            pass
        finally:
            stop_event.set()
            spot_ws.stop()
            futures_ws.stop()
            await warm.close()
            await cold.close()
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


def main() -> None:
    configure_logging(logging.INFO)
    cfg = load_config()

    try:
        asyncio.run(run_app(cfg))
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    main()
