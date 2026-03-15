"""Computation of global objects for context from pipeline state."""

from __future__ import annotations

from typing import Any

from pmai_core.domain.context_object import GlobalObjectForContext
from pmai_core.domain.tracked_object import TrackedObject
from pmai_core.resources.reid.registry import GlobalRegistry

# Type for annotated state: camera_id -> (frame, tracked_objects)
AnnotatedState = dict[str, tuple[Any, list[TrackedObject]]]


def compute_global_objects_for_context(
    annotated: AnnotatedState,
    registry: GlobalRegistry,
) -> list[GlobalObjectForContext]:
    """Build one GlobalObjectForContext per global identity (deduplicated).

    Only includes objects that have a non-empty global_id. For each identity,
    uses the view with highest confidence as representative; cameras_seen
    comes from the registry.
    """
    # Collect all (obj, camera_id) with a global_id
    by_gid: dict[str, list[tuple[TrackedObject, str]]] = {}
    for cam_id, (_frame, tracked) in annotated.items():
        for obj in tracked:
            if not obj.global_id:
                continue
            gid = obj.global_id
            if gid not in by_gid:
                by_gid[gid] = []
            by_gid[gid].append((obj, cam_id))

    result: list[GlobalObjectForContext] = []
    for gid, views in by_gid.items():
        # Representative: view with highest confidence
        best_obj, best_cam_id = max(views, key=lambda p: p[0].confidence)
        cameras_seen = sorted(registry.get_cameras_for_identity(gid))
        result.append(
            GlobalObjectForContext(
                id_global=gid,
                etiqueta=best_obj.label,
                confianza=round(best_obj.confidence, 3),
                contexto=best_obj.context,
                sensores=dict(best_obj.sensors),
                cameras_seen=cameras_seen,
                camera_id=best_cam_id,
                bbox=best_obj.bbox.to_xyxy(),
            ),
        )
    return result
