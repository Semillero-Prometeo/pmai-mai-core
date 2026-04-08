"""Domain model for the object passed to the context LLM phase.

One entry per global identity (id_global), deduplicated across cameras.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class GlobalObjectForContext(BaseModel):
    """A single re-identified object, ready for the context LLM phase.

    One per id_global; no duplicates across cameras.
    """

    id_global: str = Field(description="Global ReID identity")
    etiqueta: str = Field(description="Detection class label")
    confianza: float = Field(ge=0.0, le=1.0, description="Best confidence across views")
    contexto: str | None = Field(default=None, description="Natural-language context from LLM")
    sensores: dict[str, Any] = Field(default_factory=dict)
    cameras_seen: list[str] = Field(
        default_factory=list, description="Camera IDs that see this identity",
    )
    camera_id: str | None = Field(
        default=None, description="Representative camera (e.g. highest confidence)",
    )
    bbox: tuple[int, int, int, int] | None = Field(
        default=None,
        description="Representative bbox (xmin, ymin, xmax, ymax) from best view",
    )
    image_base64: str | None = Field(
        default=None,
        description="Base64-encoded image from best view",
    )
