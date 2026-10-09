"""Scene adapter: live objects and camera frames for one interaction turn."""

from __future__ import annotations

from typing import Any

from pmai_core.domain.context_object import GlobalObjectForContext
from pmai_core.resources.identification.service import IdentificationService


class Scene:
    def __init__(self, identification_service: IdentificationService) -> None:
        self._identification = identification_service

    def objects(self) -> list[GlobalObjectForContext]:
        return self._identification.get_global_objects_for_context()

    def frame(self, camera_id: str | None) -> Any:
        if camera_id is None:
            return None
        annotated = self._identification.get_last_annotated(camera_id)
        if annotated is None:
            return None
        ndarray, _tracked = annotated
        return ndarray
