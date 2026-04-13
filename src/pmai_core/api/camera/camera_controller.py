"""Camera listing and annotated JPEG preview over NATS (base64, no HTTP)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ValidationError

from pmai_core import __version__
from pmai_core.api.tracking_maps import build_cameras_seen_map
from pmai_core.api.image_encoding import frame_to_jpeg_base64
from pmai_core.resources.camera.manager import CameraManager
from pmai_core.resources.identification.service import IdentificationService
from pmai_core.resources.vision.overlay import draw_detections


class CameraViewPayload(BaseModel):
    camera_id: str


class AnnotatedFrameNotFoundError(Exception):
    status_code = 404


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
        cameras_seen_map = build_cameras_seen_map(svc, tracked)
        annotated = draw_detections(
            frame,
            tracked,
            cameras_seen_map=cameras_seen_map,
        )
        b64 = frame_to_jpeg_base64(annotated, max_width=None, jpeg_quality=90)
        return {
            "camera_id": camera_id,
            "media_type": "image/jpeg",
            "data_base64": b64,
        }
