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

    def get_actions(self) -> list[dict]:
        response = self.nats_handler.sync_request(
            subject=f"{self.robotics_ms}.actionService.getActions",
            payload={},
            timeout=10.0,
        )
        logger.info("robotics_actions_loaded response=%s", response)
        if isinstance(response, dict):
            data = response.get("data", [])
            actions = data if isinstance(data, list) else []
        else:
            actions = []
        logger.info("robotics_actions_loaded", count=len(actions))
        return actions

    def execute_action(self, action_id: int, arduino_id: int) -> None:
        logger.info("robotics_execute", action_id=action_id, arduino_id=arduino_id)
        self.nats_handler.sync_request(
            subject=f"{self.robotics_ms}.actionService.executeAction",
            payload={"action_id": action_id, "arduino_id": arduino_id},
            timeout=10.0,
        )
