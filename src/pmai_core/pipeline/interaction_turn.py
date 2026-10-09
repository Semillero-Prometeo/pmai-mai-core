"""One interaction turn: choose a person, caption, answer, speak, gesture."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any

from pmai_core.domain.context_object import GlobalObjectForContext
from pmai_core.pipeline.crop import crop_jpeg_base64

logger = logging.getLogger(__name__)

BUSY_PHRASE = "Sigo respondiendo, un momento."
APOLOGY_PHRASE = "Lo siento, no puedo responder en este momento."
SHORT_VOICE_PHRASE = "No te escuche con claridad, puedes repetirlo por favor."
UNIDENTIFIED_LABEL = "Persona no identificada"
MIN_VOICE_CHARS = 14
MAX_MOVEMENTS = 3


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

    def ask(self, message: str, source: str) -> str:
        stripped = message.strip()
        if source == "voice" and len(stripped) < self.min_voice_chars:
            self._speak(self._short_voice)
            return self._short_voice

        if not self._lock.acquire(blocking=False):
            if source == "voice":
                self._speak(self._busy_phrase)
            return self._busy_phrase

        try:
            return self._run_turn(stripped)
        finally:
            self._lock.release()

    def _run_turn(self, stripped: str) -> str:
        obj = self._scene_object(stripped)
        self._maybe_caption(obj)
        catalog = self._safe_catalog()
        reply, movement_ids = self._safe_answer(obj, catalog)
        rows = _execution_rows(catalog, movement_ids)
        self._speak_and_execute(reply, rows)
        return reply

    def _scene_object(self, stripped: str) -> GlobalObjectForContext:
        chosen = choose_person(self._objects())
        if chosen is None:
            return GlobalObjectForContext(
                id_global="",
                etiqueta=UNIDENTIFIED_LABEL,
                confianza=0.0,
                question=stripped,
            )
        return chosen.model_copy(update={"question": stripped})

    def _maybe_caption(self, obj: GlobalObjectForContext) -> None:
        if obj.bbox is None:
            return
        try:
            frame = self._frame(obj.camera_id)
            if frame is None:
                return
            encoded = crop_jpeg_base64(frame, obj.bbox)
            obj.image_base64 = encoded
            obj.contexto = self._caption(encoded)
        except Exception:
            logger.exception("scene_caption_failed")

    def _safe_catalog(self) -> list[dict[str, Any]]:
        try:
            catalog = self._load_catalog()
        except Exception:
            logger.exception("catalog_load_failed")
            return []
        if not isinstance(catalog, list):
            return []
        return catalog

    def _safe_answer(
        self,
        obj: GlobalObjectForContext,
        catalog: list[dict[str, Any]],
    ) -> tuple[str, list[Any]]:
        try:
            result = self._answer(obj, catalog)
            reply = str(result.get("respuesta") or "").strip()
            movimientos = result.get("movimientos") or []
            if not isinstance(movimientos, list):
                movimientos = []
        except Exception:
            logger.exception("answer_failed")
            return self._apology, []
        if not reply:
            return self._apology, []
        return reply, movimientos

    def _speak_and_execute(self, reply: str, rows: list[dict[str, Any]]) -> None:
        speak_error: list[BaseException] = []

        def _run_speak() -> None:
            try:
                self._speak(reply)
            except Exception as exc:
                speak_error.append(exc)

        speaker = threading.Thread(target=_run_speak, daemon=True, name="interaction-speak")
        speaker.start()
        for row in rows:
            try:
                self._execute(row["id"], row["arduino_id"])
            except Exception:
                logger.exception("execute_failed", extra={"action_id": row.get("id")})
        speaker.join()
        if speak_error:
            logger.error("speak_failed", exc_info=speak_error[0])


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
