import structlog

from pmai_core.core.nats.nats_client import NatsHandler, nats_handler

logger = structlog.get_logger(__name__)


class ManagementMsService:
    def __init__(self) -> None:
        self.nats_handler: NatsHandler = nats_handler
        self.management_ms: str = "MANAGEMENT_MS"

    def chat(self, message: str) -> None:
        payload = {"message": message}
        response = self.nats_handler.sync_request(
            subject=f"{self.management_ms}.chatService.chat", payload=payload, timeout=30.0
        )
        logger.info("management_chat_sent", message=message, response=response)
