from __future__ import annotations

import asyncio
from dataclasses import dataclass

from its_project.common.types import MarketData


@dataclass(slots=True, frozen=True)
class PipelineQueues:
    price_raw: asyncio.Queue[MarketData]
    lob_raw: asyncio.Queue[MarketData]
    onchain_raw: asyncio.Queue[MarketData]
    sentiment_raw: asyncio.Queue[MarketData]
    features_ready: asyncio.Queue[dict]  # feature tensors X_t^{(L)}
    predictions: asyncio.Queue[dict]  # model predictions (class, proba, confidence)
    metalearning_results: asyncio.Queue[dict]  # model selection results
    decisions: asyncio.Queue[dict]  # final trading decisions (action, size, SL/TP)
    orders: asyncio.Queue[dict]  # order execution results (order_id, status, fills)


def create_queues(maxsize: int) -> PipelineQueues:
    return PipelineQueues(
        price_raw=asyncio.Queue(maxsize=maxsize),
        lob_raw=asyncio.Queue(maxsize=maxsize),
        onchain_raw=asyncio.Queue(maxsize=maxsize),
        sentiment_raw=asyncio.Queue(maxsize=maxsize),
        features_ready=asyncio.Queue(maxsize=maxsize),
        predictions=asyncio.Queue(maxsize=maxsize),
        metalearning_results=asyncio.Queue(maxsize=maxsize),
        decisions=asyncio.Queue(maxsize=maxsize),
        orders=asyncio.Queue(maxsize=maxsize),
    )
