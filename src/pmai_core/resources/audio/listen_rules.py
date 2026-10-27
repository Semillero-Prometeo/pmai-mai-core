from __future__ import annotations

from typing import Any

import structlog

logger = structlog.get_logger(__name__)

_VAD_RATES = {8000, 16000, 32000, 48000}
_VAD_MS = {10, 20, 30}


def vad_counts_as_speech(vad: Any, chunk: bytes, sample_rate: int, chunk_ms: int) -> bool:
    if chunk_ms not in _VAD_MS or sample_rate not in _VAD_RATES:
        return True
    try:
        return bool(vad.is_speech(chunk, sample_rate))
    except Exception:
        return True


class VadSession:
    def __init__(self) -> None:
        self.logged = False

    def reset(self) -> None:
        self.logged = False

    def counts_as_speech(self, vad: Any, chunk: bytes, sample_rate: int, chunk_ms: int) -> bool:
        invalid_shape = chunk_ms not in _VAD_MS or sample_rate not in _VAD_RATES
        if invalid_shape:
            self._log_once()
            return True
        try:
            return bool(vad.is_speech(chunk, sample_rate))
        except Exception:
            self._log_once()
            return True

    def _log_once(self) -> None:
        if self.logged:
            return
        self.logged = True
        logger.warning("audio_vad_frame_unusable")


def next_silence_ms(
    silence_ms: int,
    chunk_ms: int,
    recording_ms: int,
    grace_ms: int,
    counts_as_speech: bool,
) -> int:
    if recording_ms < grace_ms or counts_as_speech:
        return 0
    return silence_ms + chunk_ms


def capture_should_end(
    recording_ms: int,
    silence_ms: int,
    min_ms: int,
    long_silence_ms: int,
    max_ms: int,
) -> bool:
    if recording_ms >= max_ms:
        return True
    return silence_ms >= long_silence_ms and recording_ms >= min_ms
