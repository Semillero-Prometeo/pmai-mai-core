"""Background audio listener with wake phrase + offline STT."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from pmai_core.resources.audio.discovery import discover_usb_microphones
from pmai_core.resources.audio.energy import WakeEnergyGate, chunk_rms
from pmai_core.resources.audio.level import LevelHold, listening_phase
from pmai_core.resources.audio.listen_rules import VadSession, capture_should_end, next_silence_ms
from pmai_core.resources.audio.models import AudioRuntimeStatus, MicrophoneInfo
from pmai_core.resources.audio.resample import FrameSlicer, resample_linear
from pmai_core.resources.audio.wake_match import WakeConfirm, match_wake_phrase
from pmai_core.settings import Settings

logger = structlog.get_logger(__name__)

try:
    import sounddevice as sd
except (ModuleNotFoundError, OSError):  # pragma: no cover - guarded at runtime
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
        self._turn: Any | None = None
        self._chat: Any | None = None
        self._speak: Any | None = None
        self._thinking_index = 0
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

        self._vosk_model: Any | None = None
        self._active_sample_rate = self._audio_settings.sample_rate
        self._selected_microphone_name: str | None = None
        self._stream_open = False
        self._capturing = False
        self._fatal = False
        self._last_chunk_rms = 0.0
        self._level = LevelHold(
            self._audio_settings.level_rms_full_scale,
            self._audio_settings.level_hold_ms,
            self._audio_settings.chunk_ms,
        )
        self._vad_session = VadSession()

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

    def bind_rpc(self, rpc_nc: Any, loop: asyncio.AbstractEventLoop) -> None:
        self._rpc_nc = rpc_nc
        self._loop = loop

    def bind_turn(self, turn: Any) -> None:
        self._turn = turn

    def bind_chat(self, chat: Any) -> None:
        self._chat = chat

    def bind_speak(self, speak: Any) -> None:
        self._speak = speak

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
        self._remember_selection(microphones)
        return microphones, self._selected_microphone_id

    def select_microphone(self, microphone_id: str) -> str:
        microphones = discover_usb_microphones(preferred_id=microphone_id)
        for mic in microphones:
            if mic.microphone_id == microphone_id:
                with self._state_lock:
                    self._selected_microphone_id = microphone_id
                    self._selected_microphone_name = mic.name
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
        now_ms = int(time.monotonic() * 1000)
        level = self._level.level_at(now_ms)
        with self._state_lock:
            phase = listening_phase(
                fatal=self._fatal,
                stream_open=self._stream_open,
                capturing=self._capturing,
                last_rms=self._last_chunk_rms,
                threshold=self._audio_settings.wake_energy_rms,
            )
            return AudioRuntimeStatus(
                running=self._running,
                listening_enabled=self._listening_enabled,
                selected_microphone_id=self._selected_microphone_id,
                selected_microphone_name=self._selected_microphone_name,
                stream_open=self._stream_open,
                phase=phase,
                input_level=level,
                wake_word_detected_at=self._wake_word_detected_at,
                last_transcript=self._last_transcript,
                last_transcript_at=self._last_transcript_at,
                last_error=self._last_error,
            )

    def choose_open_target(self, microphones: list[MicrophoneInfo]) -> MicrophoneInfo | None:
        selected = self._remember_selection(microphones)
        if selected is None:
            self._mark_disconnected()
            return None
        for mic in microphones:
            if mic.microphone_id == selected:
                with self._state_lock:
                    self._selected_microphone_name = mic.name
                return mic
        self._mark_disconnected()
        return None

    def apply_listen_failure(self, exc: Exception) -> float:
        if self._is_portaudio(exc):
            self._mark_disconnected(detail=str(exc))
            return 2.0
        self._set_fatal(str(exc))
        return 1.0

    def note_recognizer_failure(self, message: str) -> None:
        with self._state_lock:
            self._last_error = message
            self._capturing = False
            self._fatal = False
        logger.error("audio_recognizer_failed", error=message)

    def note_overflow(self) -> None:
        logger.warning("audio_input_overflow")

    def _worker_loop(self) -> None:
        while self._running and not self._stop_event.is_set():
            if not self._listening_enabled:
                self._stop_event.wait(0.5)
                continue
            try:
                self._listen_once()
            except Exception as exc:  # pragma: no cover - runtime resilience
                delay = self.apply_listen_failure(exc)
                if self._is_portaudio(exc):
                    logger.warning("audio_listener_cycle_failed", error=str(exc))
                else:
                    logger.exception("audio_listener_cycle_failed", error=str(exc))
                self._stop_event.wait(delay)

    def _listen_once(self) -> None:
        if sd is None:
            raise RuntimeError("sounddevice is not installed")
        if webrtcvad is None:
            raise RuntimeError("webrtcvad is not installed")
        if Model is None or KaldiRecognizer is None:
            raise RuntimeError("vosk is not installed")
        model = self._ensure_model()
        microphones = discover_usb_microphones(preferred_id=self._selected_microphone_id)
        target = self.choose_open_target(microphones)
        if target is None:
            self._stop_event.wait(1.0)
            return

        stream_sample_rate = self._resolve_input_sample_rate(target)
        device_chunk_frames = int(stream_sample_rate * self._audio_settings.chunk_ms / 1000)
        if device_chunk_frames <= 0:
            raise RuntimeError("Invalid audio chunk size")

        vad = webrtcvad.Vad(2)
        self._vad_session.reset()
        wake_recognizer = self._new_wake_recognizer(model)
        command_recognizer = KaldiRecognizer(model, 16000)
        slicer = FrameSlicer(480)
        gate = WakeEnergyGate(
            self._audio_settings.wake_energy_rms,
            self._audio_settings.wake_energy_hangover_ms,
        )
        confirm = WakeConfirm(self._audio_settings.wake_confirm_chunks)
        pre_roll_chunks = max(
            1,
            int(self._audio_settings.preroll_seconds * 1000 / self._audio_settings.chunk_ms),
        )
        pre_roll: deque[bytes] = deque(maxlen=pre_roll_chunks)
        silence_ms = 0
        recording_ms = 0
        active_capture = False
        detected_wake_phrase: str | None = None
        phrases = self._audio_settings.wake_phrases
        partial_match = self._audio_settings.wake_partial_match
        chunk_ms = self._audio_settings.chunk_ms

        with sd.RawInputStream(
            samplerate=stream_sample_rate,
            channels=self._audio_settings.channels,
            dtype="int16",
            blocksize=device_chunk_frames,
            device=target.device_index,
        ) as stream:
            self._mark_stream_open()
            logger.info(
                "audio_stream_opened",
                microphone_id=target.microphone_id,
                microphone_name=target.name,
                sample_rate=stream_sample_rate,
                chunk_ms=chunk_ms,
            )
            try:
                while self._running and self._listening_enabled and not self._stop_event.is_set():
                    pcm_bytes, overflowed = stream.read(device_chunk_frames)
                    if overflowed:
                        self.note_overflow()
                    samples = np.frombuffer(bytes(pcm_bytes), dtype=np.int16)
                    if stream_sample_rate != 16000:
                        samples = resample_linear(samples, stream_sample_rate, 16000)
                    for frame in slicer.push(samples):
                        try:
                            rms = chunk_rms(frame)
                            self._note_frame_rms(rms)
                            pre_roll.append(frame)
                            if not active_capture:
                                if not gate.feed(rms, chunk_ms):
                                    confirm.reset()
                                    continue
                                accepted = wake_recognizer.AcceptWaveform(frame)
                                if accepted:
                                    heard = json.loads(wake_recognizer.Result()).get("text", "")
                                    phrase = match_wake_phrase(heard, phrases, partial_match)
                                    opened = confirm.observe(phrase, True)
                                else:
                                    heard = json.loads(wake_recognizer.PartialResult()).get(
                                        "partial", ""
                                    )
                                    phrase = match_wake_phrase(heard, phrases, partial_match)
                                    opened = confirm.observe(phrase, False)
                                if not opened or phrase is None:
                                    continue
                                active_capture = True
                                self._capturing = True
                                detected_wake_phrase = phrase
                                self._mark_wake_word()
                                command_recognizer = KaldiRecognizer(model, 16000)
                                for historical in pre_roll:
                                    command_recognizer.AcceptWaveform(historical)
                                silence_ms = 0
                                recording_ms = 0
                                logger.info(
                                    "wake_word_detected", microphone_id=target.microphone_id
                                )
                                continue

                            command_recognizer.AcceptWaveform(frame)
                            recording_ms += chunk_ms
                            counts_as_speech = self._vad_session.counts_as_speech(
                                vad, frame, 16000, chunk_ms
                            )
                            silence_ms = next_silence_ms(
                                silence_ms,
                                chunk_ms,
                                recording_ms,
                                self._audio_settings.post_wake_grace_ms,
                                counts_as_speech,
                            )
                            min_ms = int(self._audio_settings.min_recording_seconds * 1000)
                            max_ms = int(self._audio_settings.max_recording_seconds * 1000)
                            if not capture_should_end(
                                recording_ms,
                                silence_ms,
                                min_ms,
                                self._audio_settings.long_silence_ms,
                                max_ms,
                            ):
                                continue
                            transcript = self._extract_transcript(command_recognizer)
                            transcript_with_wake = self._build_transcript_with_wake(
                                transcript,
                                detected_wake_phrase,
                            )
                            if transcript_with_wake:
                                self._mark_transcript(transcript_with_wake)
                                self._dispatch_transcript(transcript_with_wake)
                            else:
                                logger.info("speech_capture_finished_without_text")
                            active_capture = False
                            self._capturing = False
                            detected_wake_phrase = None
                            silence_ms = 0
                            recording_ms = 0
                            pre_roll.clear()
                            wake_recognizer = self._new_wake_recognizer(model)
                            confirm.reset()
                        except Exception as exc:
                            self.note_recognizer_failure(str(exc))
                            active_capture = False
                            detected_wake_phrase = None
                            silence_ms = 0
                            recording_ms = 0
                            pre_roll.clear()
                            wake_recognizer = self._new_wake_recognizer(model)
                            confirm.reset()
            finally:
                self._mark_stream_closed()

    def _ensure_model(self) -> Any:
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

    def _extract_transcript(self, recognizer: Any) -> str:
        final_raw = json.loads(recognizer.FinalResult()).get("text", "")
        return str(final_raw).strip()

    def _resolve_input_sample_rate(self, selected_mic: MicrophoneInfo) -> int:
        if sd is None:
            raise RuntimeError("sounddevice is not installed")
        desired = self._audio_settings.sample_rate
        candidates = [
            desired,
            selected_mic.sample_rate,
            16000,
            48000,
            44100,
            32000,
            8000,
        ]
        seen: set[int] = set()
        ordered = [sr for sr in candidates if sr > 0 and not (sr in seen or seen.add(sr))]
        last_error: str | None = None
        for sample_rate in ordered:
            try:
                sd.check_input_settings(
                    device=selected_mic.device_index,
                    samplerate=sample_rate,
                    channels=self._audio_settings.channels,
                    dtype="int16",
                )
                if sample_rate != desired:
                    logger.warning(
                        "audio_sample_rate_fallback",
                        requested_sample_rate=desired,
                        selected_sample_rate=sample_rate,
                        microphone_id=selected_mic.microphone_id,
                    )
                return sample_rate
            except Exception as exc:  # pragma: no cover - runtime dependent
                last_error = str(exc)

        raise RuntimeError(
            "No compatible sample rate for selected microphone "
            f"{selected_mic.microphone_id}. Last error: {last_error}"
        )

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

    def _build_transcript_with_wake(
        self,
        transcript: str,
        wake_phrase: str | None,
    ) -> str:
        cleaned = transcript.strip()
        if wake_phrase is None:
            return cleaned
        wake = wake_phrase.strip()
        if not cleaned:
            return wake
        lower = cleaned.lower()
        if wake in lower:
            return cleaned
        return f"{wake} {cleaned}".strip()

    def _speak_while_thinking(self, transcript: str) -> None:
        speak = self._speak
        if speak is None:
            return
        if len(transcript.strip()) < self._audio_settings.min_useful_transcript_chars:
            return
        phrases = self._audio_settings.thinking_phrases
        if not phrases:
            return
        phrase = phrases[self._thinking_index % len(phrases)]
        self._thinking_index += 1
        try:
            speak(phrase)
        except Exception as exc:
            logger.error("audio_thinking_phrase_failed", error=str(exc))

    def _dispatch_transcript(self, transcript: str) -> None:
        chat = self._chat
        if chat is None:
            logger.error("audio_chat_unbound")
            return
        self._speak_while_thinking(transcript)
        try:
            chat(transcript)
        except Exception as exc:
            logger.error("audio_voice_chat_failed", error=str(exc), exc_info=exc)

    def _remember_selection(self, microphones: list[MicrophoneInfo]) -> str | None:
        by_id = {mic.microphone_id: mic for mic in microphones}
        selected = self._selected_microphone_id
        if selected in by_id:
            with self._state_lock:
                self._selected_microphone_name = by_id[selected].name
            return selected
        if selected is not None:
            return None
        chosen: str | None = None
        preferred = self._audio_settings.preferred_microphone_id
        if preferred in by_id:
            chosen = preferred
        elif self._audio_settings.auto_select_microphone and microphones:
            chosen = microphones[0].microphone_id
        if chosen is None:
            return None
        with self._state_lock:
            self._selected_microphone_id = chosen
            self._selected_microphone_name = by_id[chosen].name
        return chosen

    def _mark_disconnected(self, detail: str | None = None) -> None:
        name = self._selected_microphone_name or "desconocido"
        message = f"Micrófono no disponible: {name}"
        if detail:
            message = f"{message}. {detail}"
        with self._state_lock:
            self._fatal = False
            self._stream_open = False
            self._capturing = False
            self._last_chunk_rms = 0.0
            self._last_error = message

    def _set_fatal(self, error: str) -> None:
        with self._state_lock:
            self._fatal = True
            self._stream_open = False
            self._capturing = False
            self._last_error = error
        logger.error("audio_listener_error", error=error)

    def _mark_stream_open(self) -> None:
        with self._state_lock:
            self._stream_open = True
            self._fatal = False
            self._last_error = None
            self._capturing = False

    def _mark_stream_closed(self) -> None:
        with self._state_lock:
            self._stream_open = False
            self._capturing = False

    def _note_frame_rms(self, rms: float) -> None:
        now_ms = int(time.monotonic() * 1000)
        self._level.update(rms, now_ms)
        with self._state_lock:
            self._last_chunk_rms = rms

    def _is_portaudio(self, exc: Exception) -> bool:
        return bool(
            sd is not None and hasattr(sd, "PortAudioError") and isinstance(exc, sd.PortAudioError)
        )

    def _new_wake_recognizer(self, model: Any) -> Any:
        return KaldiRecognizer(
            model,
            16000,
            json.dumps(self._audio_settings.wake_phrases, ensure_ascii=True),
        )

    def _sync_selection_with_hardware(self) -> None:
        microphones = discover_usb_microphones(preferred_id=self._selected_microphone_id)
        self._remember_selection(microphones)
