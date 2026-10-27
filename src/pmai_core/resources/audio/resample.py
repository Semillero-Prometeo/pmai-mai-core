from __future__ import annotations

import numpy as np


def resample_linear(samples: np.ndarray, source_rate: int, target_rate: int = 16000) -> np.ndarray:
    audio = np.asarray(samples, dtype=np.float64).reshape(-1)
    if source_rate == target_rate:
        return np.clip(np.round(audio), -32768, 32767).astype(np.int16)
    if audio.size == 0:
        return np.zeros(0, dtype=np.int16)
    out_len = int(round(audio.size * target_rate / source_rate))
    if out_len <= 1 or audio.size == 1:
        return np.clip(np.round(audio[:1]), -32768, 32767).astype(np.int16)
    positions = np.linspace(0, audio.size - 1, out_len)
    left = np.floor(positions).astype(np.int64)
    right = np.minimum(left + 1, audio.size - 1)
    frac = positions - left
    mixed = audio[left] * (1.0 - frac) + audio[right] * frac
    return np.clip(np.round(mixed), -32768, 32767).astype(np.int16)


class FrameSlicer:
    def __init__(self, frame_samples: int = 480) -> None:
        self._frame_samples = frame_samples
        self._pending = np.zeros(0, dtype=np.int16)

    @property
    def pending(self) -> np.ndarray:
        return self._pending

    def push(self, samples: np.ndarray) -> list[bytes]:
        if samples.size == 0:
            return []
        audio = np.concatenate([self._pending, np.asarray(samples, dtype=np.int16).reshape(-1)])
        frames: list[bytes] = []
        start = 0
        while start + self._frame_samples <= audio.size:
            piece = audio[start : start + self._frame_samples]
            frames.append(piece.astype(np.int16, copy=False).tobytes())
            start += self._frame_samples
        self._pending = audio[start:]
        return frames
