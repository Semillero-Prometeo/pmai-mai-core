from __future__ import annotations

import struct


def chunk_rms(pcm: bytes) -> float:
    count = len(pcm) // 2
    if count <= 0:
        return 0.0
    samples = struct.unpack(f"<{count}h", pcm[: count * 2])
    total = sum(sample * sample for sample in samples)
    return (total / count) ** 0.5


class WakeEnergyGate:
    def __init__(self, threshold: int, hangover_ms: int) -> None:
        self._threshold = threshold
        self._left_ms = 0
        self._hangover_ms = hangover_ms

    def feed(self, rms: float, chunk_ms: int) -> bool:
        if rms >= self._threshold:
            self._left_ms = self._hangover_ms
            return True
        if self._left_ms <= 0:
            return False
        self._left_ms = max(0, self._left_ms - chunk_ms)
        return True
