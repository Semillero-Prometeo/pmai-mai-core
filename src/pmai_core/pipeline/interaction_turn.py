"""One interaction turn: choose a person, caption, answer, speak, gesture."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from typing import Any

import structlog

from pmai_core.domain.context_object import GlobalObjectForContext
from pmai_core.pipeline.crop import crop_jpeg_base64

logger = structlog.get_logger(__name__)

BUSY_PHRASE = "Sigo respondiendo, un momento."
APOLOGY_PHRASE = "Lo siento, no puedo responder en este momento."
SHORT_VOICE_PHRASE = "No te escuche con claridad, puedes repetirlo por favor."
UNIDENTIFIED_LABEL = "Persona no identificada"
MIN_VOICE_CHARS = 14
MAX_MOVEMENTS = 3


class TurnResult(str):
    """Reply text that also carries where the answer came from."""

    fuente: str

    def __new__(cls, reply: str, fuente: str = "") -> TurnResult:
        value = str.__new__(cls, reply)
        value.fuente = fuente
        return value


def choose_person(
    objects: list[GlobalObjectForContext],
) -> GlobalObjectForContext | None:
    """Highest ``confianza``, then lexicographically smaller ``id_global``."""
    if not objects:
        return None
    return min(objects, key=lambda obj: (-obj.confianza, obj.id_global))


class InteractionTurn:
    """Single-flight turn that captions a crop, answers, speaks, and gestures."""

    def __init__(
        self,
        objects: Callable[[], list[GlobalObjectForContext]],
        frame: Callable[[str | None], Any],
        caption: Callable[[str], str],
        load_catalog: Callable[[], list[dict[str, Any]]],
        answer: Callable[[GlobalObjectForContext, list[dict[str, Any]]], dict[str, Any]],
        speak: Callable[[str], None],
        execute: Callable[[int, int], None],
    ) -> None:
        self._objects = objects
        self._frame = frame
        self._caption = caption
        self._load_catalog = load_catalog
        self._answer = answer
        self._speak = speak
        self._execute = execute
        self._lock = threading.Lock()
        self.min_voice_chars = MIN_VOICE_CHARS
        self._busy_phrase = BUSY_PHRASE
        self._apology = APOLOGY_PHRASE
        self._short_voice = SHORT_VOICE_PHRASE

    async def ask_async(self, message: str, source: str) -> TurnResult:
        return await asyncio.to_thread(self.ask, message, source)

    def ask(self, message: str, source: str) -> TurnResult:
        stripped = message.strip()
        logger.info("turn_start", source=source, chars=len(stripped), question=stripped[:120])
        if source == "voice" and len(stripped) < self.min_voice_chars:
            logger.info("turn_short_voice", chars=len(stripped), minimum=self.min_voice_chars)
            self._speak(self._short_voice)
            return TurnResult(self._short_voice, "voz corta")

        if not self._lock.acquire(blocking=False):
            logger.info("turn_busy", source=source)
            if source == "voice":
                self._speak(self._busy_phrase)
            return TurnResult(self._busy_phrase, "ocupado")

        try:
            return self._run_turn(stripped)
        finally:
            self._lock.release()

    def _run_turn(self, stripped: str) -> TurnResult:
        obj = self._scene_object(stripped)
        self._maybe_caption(obj)
        catalog = self._safe_catalog()
        reply, movement_ids, fuente = self._safe_answer(obj, catalog)
        rows = _execution_rows(catalog, movement_ids)
        logger.info(
            "turn_dispatch",
            fuente=fuente,
            speak=True,
            movements=[row.get("id") for row in rows],
        )
        self._speak_and_execute(reply, rows)
        logger.info("turn_done", fuente=fuente, chars=len(reply))
        return TurnResult(reply, fuente)

    def _scene_object(self, stripped: str) -> GlobalObjectForContext:
        chosen = choose_person(self._objects())
        if chosen is None:
            logger.info("turn_scene", person=UNIDENTIFIED_LABEL, has_image=False)
            return GlobalObjectForContext(
                id_global="",
                etiqueta=UNIDENTIFIED_LABEL,
                confianza=0.0,
                question=stripped,
            )
        logger.info(
            "turn_scene",
            person=chosen.etiqueta,
            id_global=chosen.id_global,
            confianza=chosen.confianza,
            camera_id=chosen.camera_id,
            has_image=chosen.bbox is not None,
        )
        return chosen.model_copy(update={"question": stripped})

    def _maybe_caption(self, obj: GlobalObjectForContext) -> None:
        if obj.bbox is None:
            logger.info("turn_caption_skipped", reason="sin_bbox")
            return
        try:
            frame = self._frame(obj.camera_id)
            if frame is None:
                logger.info("turn_caption_skipped", reason="sin_frame", camera_id=obj.camera_id)
                return
            logger.info("turn_caption_start", camera_id=obj.camera_id)
            encoded = crop_jpeg_base64(frame, obj.bbox)
            obj.image_base64 = encoded
            obj.contexto = self._caption(encoded)
            logger.info("turn_caption_done", chars=len(obj.contexto or ""))
        except Exception:
            logger.exception("turn_caption_failed")

    def _safe_catalog(self) -> list[dict[str, Any]]:
        try:
            catalog = self._load_catalog()
        except Exception:
            logger.exception("turn_catalog_failed")
            return []
        if not isinstance(catalog, list):
            logger.info("turn_catalog", count=0)
            return []
        logger.info("turn_catalog", count=len(catalog))
        return catalog

    def _safe_answer(
        self,
        obj: GlobalObjectForContext,
        catalog: list[dict[str, Any]],
    ) -> tuple[str, list[Any], str]:
        try:
            logger.info("turn_answer_start", catalog=len(catalog))
            result = self._answer(obj, catalog)
            reply = str(result.get("respuesta") or "").strip()
            movimientos = result.get("movimientos") or []
            if not isinstance(movimientos, list):
                movimientos = []
            fuente = _fuente_visible(result)
        except Exception:
            logger.exception("turn_answer_failed")
            return self._apology, [], "disculpa"
        if not reply:
            logger.info("turn_answer_empty")
            return self._apology, [], "disculpa"
        logger.info("turn_answer_done", fuente=fuente, movements=movimientos[:MAX_MOVEMENTS])
        return reply, movimientos, fuente

    def _speak_and_execute(self, reply: str, rows: list[dict[str, Any]]) -> None:
        speak_error: list[BaseException] = []

        def _run_speak() -> None:
            try:
                self._speak(reply)
            except Exception as exc:
                speak_error.append(exc)

        speaker = threading.Thread(target=_run_speak, daemon=True, name="interaction-speak")
        logger.info("turn_speak_start", chars=len(reply))
        speaker.start()
        for row in rows:
            logger.info("turn_execute", action_id=row.get("id"), arduino_id=row.get("arduino_id"))
            try:
                self._execute(row["id"], row["arduino_id"])
            except Exception:
                logger.exception("turn_execute_failed", action_id=row.get("id"))
        speaker.join()
        if speak_error:
            logger.error("turn_speak_failed", error=str(speak_error[0]))


def _fuente_visible(result: dict[str, Any]) -> str:
    fuente = str(result.get("fuente") or "")
    backend = str(result.get("backend") or result.get("_backend") or "")
    if fuente == "cache":
        return "caché"
    if backend == "openai":
        return "OpenAI"
    if fuente == "llm" or backend == "gguf":
        return "Gemma (local)"
    return fuente


def _execution_rows(
    catalog: list[dict[str, Any]],
    movement_ids: list[Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for movement_id in movement_ids[:MAX_MOVEMENTS]:
        match = _catalog_row(catalog, movement_id)
        if match is not None:
            rows.append(match)
    return rows


def _catalog_row(catalog: list[dict[str, Any]], movement_id: Any) -> dict[str, Any] | None:
    for row in catalog:
        if isinstance(row, dict) and row.get("id") == movement_id:
            return row
    return None
