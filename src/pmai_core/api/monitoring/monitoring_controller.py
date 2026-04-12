"""Object registry, trackers, global context objects, and ReID summary over NATS."""

from __future__ import annotations

from typing import Any

from pmai_core.pipeline.global_objects import compute_global_objects_for_context
from pmai_core.resources.identification.service import IdentificationService


class MonitoringController:
    def __init__(self, identification_service: IdentificationService) -> None:
        self._identification_service = identification_service

    async def get_objects(self, _: dict[str, Any]) -> dict[str, Any]:
        registry = self._identification_service.registry
        return {
            "identities": registry.size,
            "ids": registry.identity_ids,
        }

    async def get_trackers(self, _: dict[str, Any]) -> dict[str, Any]:
        return {
            "active_cameras": list(self._identification_service.trackers.keys()),
        }

    async def get_tracked_objects(self, _: dict[str, Any]) -> dict[str, Any]:
        annotated = self._identification_service.all_last_annotated
        registry = self._identification_service.registry
        global_objects = compute_global_objects_for_context(annotated, registry)
        return {
            "count": len(global_objects),
            "objects": [o.model_dump() for o in global_objects],
        }

    async def get_reid_status(self, _: dict[str, Any]) -> dict[str, Any]:
        registry = self._identification_service.registry
        identities: list[dict[str, Any]] = []
        for gid in registry.identity_ids:
            cams = registry.get_cameras_for_identity(gid)
            identities.append(
                {
                    "global_id": gid,
                    "cameras_seen": sorted(cams),
                    "cross_camera": len(cams) > 1,
                }
            )
        cross_count = sum(1 for i in identities if i["cross_camera"])
        return {
            "total_identities": len(identities),
            "cross_camera_identities": cross_count,
            "identities": identities,
        }
