"""Background audio listener with wake phrase + offline STT."""

from __future__ import annotations

import asyncio
import json
import threading
from collections import deque
from datetime import UTC, datetime
from pathlib import Path

import structlog

from pmai_core.resources.audio.discovery import discover_usb_microphones
from pmai_core.resources.audio.models import AudioRuntimeStatus, MicrophoneInfo
from pmai_core.settings import Settings

logger = structlog.get_logger(__name__)

try:
    import sounddevice as sd
except ModuleNotFoundError:  # pragma: no cover - guarded at runtime
    sd = None

try:
    import webrtcvad
except ModuleNotFoundError:  # pragma: no cover - guarded at runtime
    webrtcvad = None

try:
    from vosk import KaldiRecognizer, Model
except ModuleNotFoundError:  # pragma: no cover - guarded at runtime
    KaldiRecognizer = None  # type: ignore[assignment]
    Model = None  # type: ignore[assignment]


class AudioService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._audio_settings = settings.audio

        self._state_lock = threading.Lock()
        self._selected_microphone_id: str | None = self._audio_settings.preferred_microphone_id
        self._wake_word_detected_at: str | None = None
        self._last_transcript: str | None = None
        self._last_transcript_at: str | None = None
        self._last_error: str | None = None

        self._listening_enabled = False
        self._running = False
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None

        self._vosk_model: Model | None = None

    def start(self) -> None:
        if not self._audio_settings.enabled:
            logger.info("audio_service_disabled")
            return
        if self._running:
            return
        self._running = True
        self._stop_event.clear()
        self._listening_enabled = self._audio_settings.auto_start_listening
        self._worker = threading.Thread(
            target=self._worker_loop, daemon=True, name="audio_listener"
        )
        self._worker.start()
        logger.info("audio_service_started", auto_listening=self._listening_enabled)

    def stop(self) -> None:
        self._running = False
        self._listening_enabled = False
        self._stop_event.set()
        if self._worker is not None:
            self._worker.join(timeout=3)
            self._worker = None
        logger.info("audio_service_stopped")

    async def run_polling_loop(self) -> None:
        while self._running:
            await asyncio.sleep(self._audio_settings.poll_interval_seconds)
            self._sync_selection_with_hardware()

    def list_microphones(self) -> tuple[list[MicrophoneInfo], str | None]:
        microphones = discover_usb_microphones(preferred_id=self._selected_microphone_id)
        selected = self._ensure_selected_microphone(microphones)
        return microphones, selected

    def select_microphone(self, microphone_id: str) -> str:
        microphones = discover_usb_microphones(preferred_id=microphone_id)
        for mic in microphones:
            if mic.microphone_id == microphone_id:
                with self._state_lock:
                    self._selected_microphone_id = microphone_id
                logger.info("microphone_selected", microphone_id=microphone_id, name=mic.name)
                return microphone_id
        raise ValueError(f"Microphone not found: {microphone_id}")

    def start_listening(self) -> None:
        self._listening_enabled = True
        logger.info("audio_listening_enabled")

    def stop_listening(self) -> None:
        self._listening_enabled = False
        logger.info("audio_listening_disabled")

    def get_status(self) -> AudioRuntimeStatus:
        with self._state_lock:
            return AudioRuntimeStatus(
                running=self._running,
                listening_enabled=self._listening_enabled,
                selected_microphone_id=self._selected_microphone_id,
                wake_word_detected_at=self._wake_word_detected_at,
                last_transcript=self._last_transcript,
                last_transcript_at=self._last_transcript_at,
                last_error=self._last_error,
            )

    def _worker_loop(self) -> None:
        while self._running and not self._stop_event.is_set():
            if not self._listening_enabled:
                self._stop_event.wait(0.5)
                continue
            try:
                self._listen_once()
            except Exception as exc:  # pragma: no cover - runtime resilience
                self._set_error(str(exc))
                logger.exception("audio_listener_cycle_failed", error=str(exc))
                self._stop_event.wait(1.0)

    def _listen_once(self) -> None:
        if sd is None:
            raise RuntimeError("sounddevice is not installed")
        if webrtcvad is None:
            raise RuntimeError("webrtcvad is not installed")
        if Model is None or KaldiRecognizer is None:
            raise RuntimeError("vosk is not installed")
        model = self._ensure_model()
        microphones, selected = self.list_microphones()
        if not microphones or selected is None:
            raise RuntimeError("No microphones available for listening")
        selected_mic = next((m for m in microphones if m.microphone_id == selected), None)
        if selected_mic is None:
            raise RuntimeError(f"Selected microphone unavailable: {selected}")

        chunk_frames = int(self._audio_settings.sample_rate * self._audio_settings.chunk_ms / 1000)
        if chunk_frames <= 0:
            raise RuntimeError("Invalid audio chunk size")
        vad = webrtcvad.Vad(2)
        wake_recognizer = KaldiRecognizer(
            model,
            self._audio_settings.sample_rate,
            json.dumps(self._audio_settings.wake_phrases, ensure_ascii=True),
        )
        command_recognizer = KaldiRecognizer(model, self._audio_settings.sample_rate)

        pre_roll_chunks = max(
            1,
            int(self._audio_settings.preroll_seconds * 1000 / self._audio_settings.chunk_ms),
        )
        pre_roll = deque(maxlen=pre_roll_chunks)
        silence_ms = 0
        recording_ms = 0
        active_capture = False

        with sd.RawInputStream(
            samplerate=self._audio_settings.sample_rate,
            channels=self._audio_settings.channels,
            dtype="int16",
            blocksize=chunk_frames,
            device=selected_mic.device_index,
        ) as stream:
            logger.info(
                "audio_stream_opened",
                microphone_id=selected_mic.microphone_id,
                microphone_name=selected_mic.name,
                sample_rate=self._audio_settings.sample_rate,
                chunk_ms=self._audio_settings.chunk_ms,
            )
            while self._running and self._listening_enabled and not self._stop_event.is_set():
                pcm_bytes, _overflowed = stream.read(chunk_frames)
                chunk = bytes(pcm_bytes)
                pre_roll.append(chunk)

                if not active_capture:
                    _ = wake_recognizer.AcceptWaveform(chunk)
                    partial = json.loads(wake_recognizer.PartialResult()).get("partial", "").lower()
                    final_text = json.loads(wake_recognizer.Result()).get("text", "").lower()
                    if self._contains_wake_phrase(partial) or self._contains_wake_phrase(
                        final_text
                    ):
                        active_capture = True
                        self._mark_wake_word()
                        command_recognizer = KaldiRecognizer(
                            model, self._audio_settings.sample_rate
                        )
                        for historical in pre_roll:
                            command_recognizer.AcceptWaveform(historical)
                        silence_ms = 0
                        recording_ms = 0
                        logger.info("wake_word_detected", microphone_id=selected_mic.microphone_id)
                    continue

                command_recognizer.AcceptWaveform(chunk)
                recording_ms += self._audio_settings.chunk_ms
                if self._is_speech(chunk, vad):
                    silence_ms = 0
                else:
                    silence_ms += self._audio_settings.chunk_ms

                reached_max = recording_ms >= int(self._audio_settings.max_recording_seconds * 1000)
                reached_long_silence = silence_ms >= self._audio_settings.long_silence_ms
                enough_audio = recording_ms >= int(
                    self._audio_settings.min_recording_seconds * 1000
                )
                if reached_max or (reached_long_silence and enough_audio):
                    transcript = self._extract_transcript(command_recognizer)
                    if transcript:
                        self._mark_transcript(transcript)
                    else:
                        logger.info("speech_capture_finished_without_text")
                    active_capture = False
                    silence_ms = 0
                    recording_ms = 0
                    pre_roll.clear()
                    wake_recognizer = KaldiRecognizer(
                        model,
                        self._audio_settings.sample_rate,
                        json.dumps(self._audio_settings.wake_phrases, ensure_ascii=True),
                    )

    def _ensure_model(self) -> Model:
        if Model is None:
            raise RuntimeError("vosk is not installed")
        if self._vosk_model is not None:
            return self._vosk_model
        model_path = Path(self._audio_settings.vosk_model_path)
        if not model_path.exists():
            raise RuntimeError(f"Vosk model not found: {model_path}")
        self._vosk_model = Model(str(model_path))
        logger.info("vosk_model_loaded", model_path=str(model_path))
        return self._vosk_model

    def _contains_wake_phrase(self, text: str) -> bool:
        value = text.strip().lower()
        if not value:
            return False
        for wake in self._audio_settings.wake_phrases:
            if wake in value:
                return True
            if not self._audio_settings.wake_partial_match and value == wake:
                return True
        return False

    def _extract_transcript(self, recognizer: KaldiRecognizer) -> str:
        final_raw = json.loads(recognizer.FinalResult()).get("text", "")
        return str(final_raw).strip()

    def _is_speech(self, chunk: bytes, vad: webrtcvad.Vad) -> bool:
        chunk_ms = self._audio_settings.chunk_ms
        if chunk_ms not in (10, 20, 30):
            return False
        return bool(vad.is_speech(chunk, self._audio_settings.sample_rate))

    def _mark_wake_word(self) -> None:
        with self._state_lock:
            self._wake_word_detected_at = datetime.now(UTC).isoformat()
            self._last_error = None

    def _mark_transcript(self, transcript: str) -> None:
        now = datetime.now(UTC).isoformat()
        with self._state_lock:
            self._last_transcript = transcript
            self._last_transcript_at = now
            self._last_error = None
        logger.info("speech_transcribed", transcript=transcript)

    def _set_error(self, error: str) -> None:
        with self._state_lock:
            self._last_error = error
        logger.error("audio_listener_error", error=error)

    def _ensure_selected_microphone(self, microphones: list[MicrophoneInfo]) -> str | None:
        if not microphones:
            with self._state_lock:
                self._selected_microphone_id = None
            return None

        selected = self._selected_microphone_id
        available_ids = {mic.microphone_id for mic in microphones}
        if selected in available_ids:
            return selected

        chosen: str | None = None
        if self._audio_settings.preferred_microphone_id in available_ids:
            chosen = self._audio_settings.preferred_microphone_id
        elif self._audio_settings.auto_select_microphone:
            chosen = microphones[0].microphone_id
        with self._state_lock:
            self._selected_microphone_id = chosen
        return chosen

    def _sync_selection_with_hardware(self) -> None:
        microphones = discover_usb_microphones(preferred_id=self._selected_microphone_id)
        self._ensure_selected_microphone(microphones)
        # Lightweight heartbeat to aid observability.
        if microphones:
            logger.debug(
                "audio_microphones_polled",
                count=len(microphones),
                selected_microphone_id=self._selected_microphone_id,
            )
