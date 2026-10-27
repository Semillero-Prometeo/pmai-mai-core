from __future__ import annotations

from pmai_core.resources.audio.models import ListenPhase


class LevelHold:
    def __init__(self, full_scale: int, hold_ms: int, chunk_ms: int) -> None:
        self._full_scale = max(1, full_scale)
        self._hold_ms = hold_ms
        self._chunk_ms = chunk_ms
        self._samples: list[tuple[int, float]] = []

    def update(self, rms: float, now_ms: int) -> int:
        self._samples.append((now_ms, rms))
        return self.level_at(now_ms)

    def level_at(self, now_ms: int) -> int:
        if not self._samples:
            return 0
        newest = self._samples[-1][0]
        window_start = newest - self._hold_ms
        peak = max(rms for stamp, rms in self._samples if stamp >= window_start)
        self._samples = [(stamp, rms) for stamp, rms in self._samples if stamp >= window_start]
        scaled = self._scale(peak)
        gap = now_ms - newest
        if gap <= self._chunk_ms:
            return scaled
        elapsed = gap - self._chunk_ms
        if elapsed >= self._hold_ms:
            return 0
        return int(round(scaled * (1 - elapsed / self._hold_ms)))

    def _scale(self, rms: float) -> int:
        return max(0, min(100, int(round(rms * 100 / self._full_scale))))


def listening_phase(
    *,
    fatal: bool,
    stream_open: bool,
    capturing: bool,
    last_rms: float,
    threshold: float,
) -> ListenPhase:
    if fatal:
        return ListenPhase.ERROR
    if not stream_open:
        return ListenPhase.DISCONNECTED
    if capturing:
        return ListenPhase.CAPTURING
    if last_rms >= threshold:
        return ListenPhase.HEARING
    return ListenPhase.SILENT
