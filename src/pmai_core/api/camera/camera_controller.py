"""Camera listing and annotated JPEG preview over NATS (base64, no HTTP)."""

from __future__ import annotations

import base64
from typing import Any

import cv2
from pydantic import BaseModel, ValidationError

from pmai_core import __version__
from pmai_core.resources.camera.manager import CameraManager
from pmai_core.resources.identification.service import IdentificationService
from pmai_core.resources.vision.overlay import draw_detections


class CameraViewPayload(BaseModel):
    camera_id: str


class AnnotatedFrameNotFoundError(Exception):
    status_code = 404


def _build_cameras_seen_map(
    identification_service: IdentificationService,
    tracked: list[Any],
) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for obj in tracked:
        gid = obj.global_id
        if gid and gid not in result:
            result[gid] = identification_service.registry.get_cameras_for_identity(gid)
    return result


class CameraController:
    def __init__(
        self,
        camera_manager: CameraManager,
        identification_service: IdentificationService,
    ) -> None:
        self._camera_manager = camera_manager
        self._identification_service = identification_service

    async def get_cameras(self, _: dict[str, Any]) -> dict[str, Any]:
        infos = self._camera_manager.cameras
        return {
            "version": __version__,
            "count": len(infos),
            "cameras": [
                {
                    "camera_id": info.camera_id,
                    "source_type": info.source_type.value,
                    "device_path": info.device_path,
                    "name": info.name,
                    "resolution": list(info.resolution),
                    "fps": info.fps,
                    "status": info.status.value,
                }
                for info in infos.values()
            ],
        }

    async def get_camera_view(self, data: dict[str, Any]) -> dict[str, Any]:
        try:
            payload = CameraViewPayload.model_validate(data)
        except ValidationError as e:
            raise ValueError(str(e)) from e
        camera_id = payload.camera_id
        svc = self._identification_service
        frame_data = svc.get_last_annotated(camera_id)
        if frame_data is None:
            raise AnnotatedFrameNotFoundError(
                f"No annotated frame for camera {camera_id}",
            )
        frame, tracked = frame_data
        cameras_seen_map = _build_cameras_seen_map(svc, tracked)
        annotated = draw_detections(
            frame,
            tracked,
            cameras_seen_map=cameras_seen_map,
        )
        _, jpeg = cv2.imencode(".jpg", annotated)
        return {
            "camera_id": camera_id,
            "media_type": "image/jpeg",
            "data_base64": base64.b64encode(jpeg.tobytes()).decode("ascii"),
        }
