"""NATS request–reply handler aligned with the NestJS gateway envelope (see ms-robotics)."""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
from typing import Any

import structlog
from nats.aio.client import Client as NATSClient
from nats.aio.errors import ErrNoServers
from nats.aio.msg import Msg as NATSMessage

from pmai_core.core.nats.interfaces.nats_interface import NatsSubscriber

logger = structlog.get_logger(__name__)


def _rpc_error_payload(exc: BaseException) -> str | dict[str, Any]:
    """Nest gateway maps ``{ statusCode, message }`` to HTTP; plain strings become 400."""
    code = getattr(exc, "status_code", None)
    if isinstance(code, int) and 100 <= code <= 599:
        return {"statusCode": code, "message": str(exc)}
    return str(exc)


class NatsHandler:
    _rpc_nc: Any | None = None
    _loop: asyncio.AbstractEventLoop | None = None

    def __init__(self) -> None:
        self.nc: NATSClient = NATSClient()
        self.connected: bool = False
        self._rpc_nc = None
        self._loop = None

    async def connect(self, url: str, *, max_attempts: int = 10) -> None:
        try:
            logger.info("nats_rpc_connecting", url=url)
            for attempt in range(max_attempts):
                try:
                    await self.nc.connect(
                        servers=[url],
                        connect_timeout=10,
                        reconnect_time_wait=2,
                        max_reconnect_attempts=10,
                    )
                    if self.nc.is_connected:
                        self.connected = True
                        self._rpc_nc = self.nc
                        self._loop = asyncio.get_running_loop()
                        logger.info("nats_rpc_connected", url=url)
                        return
                except Exception as e:
                    logger.warning(
                        "nats_rpc_connect_retry",
                        attempt=attempt + 1,
                        max_attempts=max_attempts,
                        error=str(e),
                    )
                    await asyncio.sleep(2)
            raise RuntimeError("Could not connect to NATS for RPC after multiple attempts")
        except ErrNoServers as e:
            logger.error("nats_rpc_no_servers", error=str(e))
            self.connected = False
            raise
        except Exception as e:
            logger.error("nats_rpc_connect_failed", error=str(e))
            self.connected = False
            raise

    async def disconnect(self) -> None:
        if self.connected:
            await self.nc.close()
            self.connected = False
            logger.info("nats_rpc_disconnected")

    async def subscribe(self, subscriber: NatsSubscriber) -> None:
        async def message_handler(msg: NATSMessage) -> None:
            try:
                if msg.reply:
                    data = json.loads(msg.data.decode())
                    response_data = await subscriber.controller(data["data"])
                    if hasattr(response_data, "model_dump"):
                        response_payload = response_data.model_dump()
                    elif hasattr(response_data, "dict"):
                        response_payload = response_data.dict()
                    else:
                        response_payload = response_data

                    await msg.respond(
                        json.dumps({"response": response_payload, "isDisposed": True}).encode()
                    )
                else:
                    data = json.loads(msg.data.decode())
                    await subscriber.controller(data)

            except json.JSONDecodeError as e:
                logger.error("nats_rpc_invalid_json", error=str(e))
                if msg.reply:
                    await msg.respond(
                        json.dumps({"err": "Invalid JSON", "isDisposed": True}).encode()
                    )
            except Exception as e:
                logger.exception("nats_rpc_handler_error", error=str(e))
                if msg.reply:
                    err_body = _rpc_error_payload(e)
                    await msg.respond(json.dumps({"err": err_body, "isDisposed": True}).encode())

        logger.info("nats_rpc_subscribing", subject=subscriber.subject)
        await self.nc.subscribe(subscriber.subject, cb=message_handler)

    def sync_request(
        self,
        subject: str,
        payload: dict[str, Any],
        timeout: float,
    ) -> dict[str, Any]:
        if self._loop is None or self._rpc_nc is None:
            raise RuntimeError("NATS RPC not available for audio dispatch")
        request_payload = {"id": "audio-voice-dispatch", "data": payload}
        future = asyncio.run_coroutine_threadsafe(
            self._rpc_nc.request(subject, json.dumps(request_payload).encode(), timeout=timeout),
            self._loop,
        )
        try:
            msg = future.result(timeout=timeout + 2.0)
        except concurrent.futures.TimeoutError as exc:
            raise RuntimeError(f"NATS request timeout for {subject}") from exc

        raw_data = json.loads(msg.data.decode())
        if "err" in raw_data and raw_data["err"]:
            raise RuntimeError(f"NATS error for {subject}: {raw_data['err']}")
        response_data = raw_data.get("response")
        if isinstance(response_data, dict):
            return response_data
        return {"response": response_data}


nats_handler: NatsHandler = NatsHandler()
