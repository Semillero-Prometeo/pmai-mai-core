import structlog

from pmai_core.core.nats.nats_client import NatsHandler, nats_handler

logger = structlog.get_logger(__name__)


class RoboticsMsService:
    def __init__(self) -> None:
        self.nats_handler: NatsHandler = nats_handler
        self.robotics_ms: str = "ROBOTICS_MS"

    def speak(self, message: str) -> None:
        payload = {"message": message}
        response = self.nats_handler.sync_request(
            subject=f"{self.robotics_ms}.voiceService.speak", payload=payload, timeout=30.0
        )
        logger.info("audio_robotics_speak_sent", message=message, response=response)
