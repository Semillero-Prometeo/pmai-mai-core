import asyncio

from pmai_core.api.interaction.interaction_controller import InteractionController
from pmai_core.api.robotics_ms.robotics_ms_service import RoboticsMsService
from pmai_core.resources.audio.service import AudioService
from pmai_core.settings import Settings


class FakeNats:
    def __init__(self):
        self.calls = []

    def sync_request(self, subject, payload, timeout):
        self.calls.append((subject, payload, timeout))
        if subject.endswith("getActions"):
            return {"total": 1, "data": [{"id": 3, "name": "Negar", "arduino_id": 2, "description": ""}]}
        return {"status": "ok", "response": []}


def test_robotics_client_reads_actions_and_executes():
    service = RoboticsMsService()
    service.nats_handler = FakeNats()
    assert service.get_actions()[0]["id"] == 3
    service.execute_action(3, 2)
    assert service.nats_handler.calls[1][0].endswith("actionService.executeAction")
    assert service.nats_handler.calls[1][1] == {"action_id": 3, "arduino_id": 2}


def test_ask_controller_uses_chat_source():
    seen = {}

    class Turn:
        def ask(self, message: str, source: str) -> str:
            seen["message"] = message
            seen["source"] = source
            return "listo"

    controller = InteractionController(Turn())
    result = asyncio.run(controller.ask({"message": "hola desde la web"}))
    assert result.reply == "listo"
    assert seen == {"message": "hola desde la web", "source": "chat"}


class FakeFuture:
    def __init__(self):
        self.callbacks = []

    def add_done_callback(self, callback):
        self.callbacks.append(callback)

    def cancelled(self):
        return False

    def exception(self):
        return None


def test_voice_dispatch_schedules_and_returns(monkeypatch):
    scheduled = {}

    def fake_submit(coro, loop):
        scheduled["coro"] = coro
        coro.close()
        future = FakeFuture()
        scheduled["future"] = future
        return future

    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", fake_submit)
    settings = Settings()
    audio = AudioService(settings)

    class Turn:
        async def ask_async(self, message: str, source: str) -> str:
            return "voz"

    audio.bind_turn(Turn())
    audio._loop = object()
    audio._dispatch_transcript("esta transcripcion si es larga")
    assert scheduled["coro"] is not None
    assert scheduled["future"].callbacks


def test_voice_dispatch_logs_future_exception(monkeypatch):
    logged = {}
    captured = {}

    class FailingFuture(FakeFuture):
        def exception(self):
            return RuntimeError("boom")

    def fake_submit(coro, loop):
        coro.close()
        future = FailingFuture()
        captured["future"] = future
        return future

    def fake_error(event, **kwargs):
        logged["event"] = event
        logged["error"] = kwargs.get("error")

    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", fake_submit)
    monkeypatch.setattr("pmai_core.resources.audio.service.logger.error", fake_error)
    audio = AudioService(Settings())

    class Turn:
        async def ask_async(self, message: str, source: str) -> str:
            return "voz"

    audio.bind_turn(Turn())
    audio._loop = object()
    audio._dispatch_transcript("esta transcripcion si es larga")
    captured["future"].callbacks[0](captured["future"])
    assert logged["event"] == "audio_voice_turn_failed"
    assert logged["error"] == "boom"
