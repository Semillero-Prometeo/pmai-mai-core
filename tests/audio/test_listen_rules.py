from __future__ import annotations

from pmai_core.resources.audio.listen_rules import (
    VadSession,
    capture_should_end,
    next_silence_ms,
    vad_counts_as_speech,
)


class _Boom:
    def is_speech(self, _chunk: bytes, _rate: int) -> bool:
        raise RuntimeError("bad frame")


class _Speech:
    def is_speech(self, _chunk: bytes, _rate: int) -> bool:
        return True


def test_invalid_vad_frame_counts_as_speech() -> None:
    assert vad_counts_as_speech(_Boom(), b"\x00\x00", 16000, 30) is True
    assert vad_counts_as_speech(_Speech(), b"\x00\x00", 44100, 30) is True


def test_vad_session_logs_once() -> None:
    session = VadSession()
    assert session.counts_as_speech(_Boom(), b"\x00\x00", 16000, 30) is True
    assert session.logged is True
    assert session.counts_as_speech(_Boom(), b"\x00\x00", 16000, 30) is True
    session.reset()
    assert session.logged is False


def test_invalid_frame_does_not_add_silence() -> None:
    assert next_silence_ms(1500, 30, 3000, 2200, True) == 0


def test_silence_grows_only_after_grace_and_nonspeech() -> None:
    assert next_silence_ms(0, 30, 1000, 2200, False) == 0
    assert next_silence_ms(1500, 30, 3000, 2200, False) == 1530


def test_short_pause_does_not_end_capture() -> None:
    assert capture_should_end(3000, 1500, 1000, 1600, 20000) is False
    assert capture_should_end(3000, 1600, 1000, 1600, 20000) is True
    assert capture_should_end(500, 1600, 1000, 1600, 20000) is False
    assert capture_should_end(20000, 0, 1000, 1600, 20000) is True
