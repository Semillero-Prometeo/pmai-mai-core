from __future__ import annotations

from pmai_core.api.audio.audio_controller import listening_status_payload
from pmai_core.resources.audio.models import AudioRuntimeStatus, ListenPhase


def test_payload_includes_live_fields() -> None:
    status = AudioRuntimeStatus(
        running=True,
        listening_enabled=True,
        selected_microphone_id="abc",
        selected_microphone_name="USB Mic",
        stream_open=True,
        phase=ListenPhase.HEARING,
        input_level=40,
        wake_word_detected_at=None,
        last_transcript=None,
        last_transcript_at=None,
        last_error=None,
    )
    payload = listening_status_payload(status)
    assert payload["selected_microphone_name"] == "USB Mic"
    assert payload["stream_open"] is True
    assert payload["phase"] == "hearing"
    assert payload["input_level"] == 40
    assert payload["running"] is True
    assert payload["last_transcript"] is None
