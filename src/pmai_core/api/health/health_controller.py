"""Health probe for NATS RPC (same envelope as ms-robotics ``AppController.health``)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pmai_core.core.interfaces.health import Health
from pmai_core.settings import Settings


class HealthController:
    def __init__(self, settings: Settings) -> None:
        self._ms_name = settings.nats.ms_name

    async def health(self, _: dict[str, Any]) -> Health:
        return Health(
            status="UP",
            timestamp=datetime.now().isoformat(),
            msName=self._ms_name,
        )
