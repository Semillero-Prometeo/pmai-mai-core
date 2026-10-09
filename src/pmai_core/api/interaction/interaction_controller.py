"""NATS controller that runs one chat turn on a worker thread."""

from __future__ import annotations

import asyncio
from typing import Any

from pydantic import BaseModel


class AskReply(BaseModel):
    reply: str


class InteractionController:
    def __init__(self, turn) -> None:
        self._turn = turn

    async def ask(self, data: dict[str, Any]) -> AskReply:
        message = str(data.get("message") or "")
        reply = await asyncio.to_thread(self._turn.ask, message, "chat")
        return AskReply(reply=reply)
