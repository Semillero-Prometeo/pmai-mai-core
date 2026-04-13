"""Aggregated vision snapshot: cameras (thumbnails + optional preview), objects, ReID."""

from __future__ import annotations

import time
from typing import Any

from pydantic import ValidationError

from pmai_core import __version__
from pmai_core.api.image_encoding import frame_to_jpeg_base64
from pmai_core.api.tracking_maps import build_cameras_seen_map
from pmai_core.api.vision.schemas import (
    CameraStreamTile,
    ReidIdentitySummary,
    ReidSummary,
    VisionSnapshotRequest,
    VisionSnapshotResponse,
)
from pmai_core.pipeline.global_objects import compute_global_objects_for_context
from pmai_core.resources.camera.manager import CameraManager
from pmai_core.resources.identification.service import IdentificationService
from pmai_core.resources.vision.overlay import draw_detections


class VisionController:
    def __init__(
        self,
        camera_manager: CameraManager,
        identification_service: IdentificationService,
    ) -> None:
        self._camera_manager = camera_manager
        self._identification_service = identification_service

    async def get_snapshot(self, data: dict[str, Any]) -> dict[str, Any]:
        try:
            req = VisionSnapshotRequest.model_validate(data or {})
        except ValidationError as e:
            raise ValueError(str(e)) from e

        svc = self._identification_service
        registry = svc.registry
        annotated = svc.all_last_annotated
        global_objects = compute_global_objects_for_context(annotated, registry)

        identities: list[ReidIdentitySummary] = []
        for gid in registry.identity_ids:
            cams = registry.get_cameras_for_identity(gid)
            identities.append(
                ReidIdentitySummary(
                    global_id=gid,
                    cameras_seen=sorted(cams),
                    cross_camera=len(cams) > 1,
                )
            )
        cross_count = sum(1 for i in identities if i.cross_camera)
        reid_summary = ReidSummary(
            total_identities=len(identities),
            cross_camera_identities=cross_count,
            identities=identities,
        )

        tiles: list[CameraStreamTile] = []
        for info in self._camera_manager.cameras.values():
            frame_data = svc.get_last_annotated(info.camera_id)
            if frame_data is None:
                tiles.append(
                    CameraStreamTile(
                        camera_id=info.camera_id,
                        source_type=info.source_type.value,
                        device_path=info.device_path,
                        name=info.name,
                        resolution=list(info.resolution),
                        fps=info.fps,
                        status=info.status.value,
                        has_frame=False,
                        thumbnail_jpeg_base64=None,
                        preview_jpeg_base64=None,
                    )
                )
                continue

            frame, tracked = frame_data
            cameras_seen_map = build_cameras_seen_map(svc, tracked)
            annotated_frame = draw_detections(
                frame,
                tracked,
                cameras_seen_map=cameras_seen_map,
            )

            thumb_b64 = frame_to_jpeg_base64(
                annotated_frame,
                max_width=req.thumb_max_width,
                jpeg_quality=82,
            )
            preview_b64: str | None = None
            if req.selected_camera_id is not None and req.selected_camera_id == info.camera_id:
                preview_b64 = frame_to_jpeg_base64(
                    annotated_frame,
                    max_width=req.preview_max_width,
                    jpeg_quality=88,
                )

            tiles.append(
                CameraStreamTile(
                    camera_id=info.camera_id,
                    source_type=info.source_type.value,
                    device_path=info.device_path,
                    name=info.name,
                    resolution=list(info.resolution),
                    fps=info.fps,
                    status=info.status.value,
                    has_frame=True,
                    thumbnail_jpeg_base64=thumb_b64,
                    preview_jpeg_base64=preview_b64,
                )
            )

        response = VisionSnapshotResponse(
            schema_version=1,
            timestamp_ms=int(time.time() * 1000),
            version=__version__,
            cameras=tiles,
            global_objects=[o.model_dump(mode="json") for o in global_objects],
            reid_summary=reid_summary,
        )
        return response.model_dump(mode="json", by_alias=True)
