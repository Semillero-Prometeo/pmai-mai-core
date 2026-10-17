from pmai_core.resources.audio.service import AudioService
from pmai_core.settings import Settings


def test_dispatch_transcript_calls_bound_chat_not_turn() -> None:
    calls: list[str] = []
    turn_calls: list[str] = []

    class FakeTurn:
        def ask_async(self, transcript: str, source: str) -> None:
            turn_calls.append(f"{source}:{transcript}")

    service = AudioService(Settings())
    service.bind_chat(calls.append)
    service.bind_turn(FakeTurn())
    service._dispatch_transcript("hey fede hola")
    assert calls == ["hey fede hola"]
    assert turn_calls == []
