"""Helpers for ReID overlay maps shared by camera and vision APIs."""

from __future__ import annotations

from typing import Any

from pmai_core.resources.identification.service import IdentificationService


def build_cameras_seen_map(
    identification_service: IdentificationService,
    tracked: list[Any],
) -> dict[str, list[str]]:
    """Build {global_id: [cameras...]} for objects in the given tracked list."""
    result: dict[str, list[str]] = {}
    for obj in tracked:
        gid = obj.global_id
        if gid and gid not in result:
            result[gid] = identification_service.registry.get_cameras_for_identity(gid)
    return result
