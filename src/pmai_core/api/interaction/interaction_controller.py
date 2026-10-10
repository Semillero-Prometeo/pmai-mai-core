"""NATS controller that runs one chat turn on a worker thread."""

from __future__ import annotations

import asyncio
from typing import Any

from pydantic import BaseModel


class ExecutedMovement(BaseModel):
    id: int
    name: str


class AskReply(BaseModel):
    reply: str
    fuente: str = ""
    movimientos: list[ExecutedMovement] = []


class InteractionController:
    def __init__(self, turn) -> None:
        self._turn = turn

    async def ask(self, data: dict[str, Any]) -> AskReply:
        message = str(data.get("message") or "")
        reply = await asyncio.to_thread(self._turn.ask, message, "chat")
        raw_movements = getattr(reply, "movimientos", []) or []
        movements = [
            ExecutedMovement(id=int(item["id"]), name=str(item.get("name") or item["id"]))
            for item in raw_movements
            if isinstance(item, dict) and "id" in item
        ]
        return AskReply(
            reply=str(reply),
            fuente=str(getattr(reply, "fuente", "")),
            movimientos=movements,
        )
