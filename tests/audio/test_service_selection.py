from __future__ import annotations

from pmai_core.resources.audio.discovery import build_fingerprint
from pmai_core.resources.audio.models import ListenPhase, MicrophoneInfo
from pmai_core.resources.audio.service import AudioService
from pmai_core.settings import Settings


def _mic(name: str, index: int) -> MicrophoneInfo:
    fingerprint = build_fingerprint(name, "ALSA", 1)
    return MicrophoneInfo(
        microphone_id=fingerprint,
        name=name,
        hostapi_name="ALSA",
        device_index=index,
        hardware_fingerprint=fingerprint,
        channels=1,
        sample_rate=16000,
    )


def test_same_fingerprint_follows_new_index() -> None:
    service = AudioService(Settings())
    first = _mic("USB Mic", 0)
    service.choose_open_target([first])
    again = _mic("USB Mic", 7)
    chosen = service.choose_open_target([again])
    assert chosen is not None
    assert chosen.device_index == 7
    assert chosen.microphone_id == first.microphone_id


def test_missing_mic_stays_selected_and_disconnected() -> None:
    service = AudioService(Settings())
    service.start_listening()
    usb = _mic("USB Mic", 0)
    service.choose_open_target([usb])
    other = _mic("USB Other", 1)
    assert service.choose_open_target([other]) is None
    status = service.get_status()
    assert status.selected_microphone_id == usb.microphone_id
    assert status.selected_microphone_name == "USB Mic"
    assert status.phase == ListenPhase.DISCONNECTED
    assert status.listening_enabled is True
    assert status.last_error == "Micrófono no disponible: USB Mic"


def test_runtime_error_is_phase_error_and_waits_one_second() -> None:
    service = AudioService(Settings())
    delay = service.apply_listen_failure(RuntimeError("Vosk model not found"))
    status = service.get_status()
    assert delay == 1.0
    assert status.phase == ListenPhase.ERROR
    assert status.last_error == "Vosk model not found"


def test_portaudio_failure_is_disconnected_and_waits_two_seconds(monkeypatch) -> None:
    service = AudioService(Settings())
    service.choose_open_target([_mic("USB Mic", 0)])
    monkeypatch.setattr(service, "_is_portaudio", lambda _exc: True)
    delay = service.apply_listen_failure(RuntimeError("device gone"))
    status = service.get_status()
    assert delay == 2.0
    assert status.phase == ListenPhase.DISCONNECTED
    assert status.selected_microphone_name == "USB Mic"


def test_recognizer_failure_keeps_the_stream_open() -> None:
    service = AudioService(Settings())
    service._stream_open = True
    service._fatal = False
    service._last_chunk_rms = 100
    service.note_recognizer_failure("bad json")
    status = service.get_status()
    assert status.stream_open is True
    assert status.phase == ListenPhase.SILENT
    assert status.last_error == "bad json"


def test_overflow_does_not_change_phase() -> None:
    service = AudioService(Settings())
    service._stream_open = True
    service._fatal = False
    service._last_chunk_rms = 800
    service.note_overflow()
    status = service.get_status()
    assert status.phase == ListenPhase.HEARING
    assert status.last_error is None


def _boom(_text: str) -> None:
    raise RuntimeError("turn down")


def test_dispatch_failure_keeps_the_transcript() -> None:
    service = AudioService(Settings())
    service._stream_open = True
    service._fatal = False
    service._last_chunk_rms = 100
    service._chat = _boom
    service._mark_transcript("hey fede hola")
    service._dispatch_transcript("hey fede hola")
    service._capturing = False
    status = service.get_status()
    assert status.last_transcript == "hey fede hola"
    assert status.phase == ListenPhase.SILENT
