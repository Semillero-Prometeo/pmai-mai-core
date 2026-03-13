"""Core domain entity representing a tracked and re-identified object."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from pmai_core.domain.detection import BBox


class TrackedObject(BaseModel):
    """An object being tracked across one or more cameras.

    Fields align with the project-wide object schema:
        { id, id_global (ReID), Etiqueta, Contexto, Sensores }
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: int = Field(description="Local tracking ID (per-camera)")
    global_id: str = Field(default="", description="Global ReID identity")
    label: str = Field(description="Detection class label")
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: BBox
    camera_id: str = Field(description="Source camera identifier")
    context: str | None = Field(default=None, description="Natural-language context from LLM")
    sensors: dict[str, Any] = Field(default_factory=dict)
    embedding: NDArray[np.float32] | None = Field(default=None, description="ReID feature vector")
    last_seen: float = Field(default_factory=time.time)
