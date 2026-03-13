"""Async NATS client for publishing and subscribing to pipeline events."""

from __future__ import annotations

import json
from collections.abc import Callable, Coroutine
from typing import Any

import nats
import structlog
from nats.aio.client import Client as NATSConnection

from pmai_core.settings import NATSSettings

logger = structlog.get_logger(__name__)


class NATSClient:
    """Thin wrapper around ``nats-py`` for structured pub/sub.

    Subjects are automatically prefixed with the configured
    ``subject_prefix`` (e.g. ``pmai.detection``).
    """

    def __init__(self, settings: NATSSettings) -> None:
        self._url = settings.url
        self._prefix = settings.subject_prefix
        self._nc: NATSConnection | None = None

    @property
    def is_connected(self) -> bool:
        return self._nc is not None and self._nc.is_connected

    async def connect(self) -> None:
        try:
            self._nc = await nats.connect(self._url)
            logger.info("nats_connected", url=self._url)
        except Exception:
            logger.warning("nats_connection_failed", url=self._url)
            self._nc = None

    async def close(self) -> None:
        if self._nc is not None:
            await self._nc.close()
            logger.info("nats_disconnected")

    async def publish(self, subject_suffix: str, data: dict[str, Any]) -> None:
        """Publish a JSON-serialised message to ``{prefix}.{subject_suffix}``."""
        if self._nc is None or not self._nc.is_connected:
            return
        subject = f"{self._prefix}.{subject_suffix}"
        payload = json.dumps(data, default=str).encode()
        await self._nc.publish(subject, payload)

    async def subscribe(
        self,
        subject_suffix: str,
        handler: Callable[[dict[str, Any]], Coroutine[Any, Any, None]],
    ) -> None:
        """Subscribe to ``{prefix}.{subject_suffix}`` and dispatch decoded messages."""
        if self._nc is None:
            return
        subject = f"{self._prefix}.{subject_suffix}"

        async def _cb(msg: Any) -> None:
            data = json.loads(msg.data.decode())
            await handler(data)

        await self._nc.subscribe(subject, cb=_cb)
        logger.info("nats_subscribed", subject=subject)
