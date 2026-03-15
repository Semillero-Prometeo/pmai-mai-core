"""FastAPI router with health, camera, and object monitoring endpoints."""

from __future__ import annotations

from typing import Any

import cv2
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response

from pmai_core import __version__
from pmai_core.vision.overlay import draw_detections
from pmai_core.camera.manager import CameraManager
from pmai_core.pipeline.engine import PipelineEngine


def create_api(
    camera_manager: CameraManager | None = None,
    pipeline_engine: PipelineEngine | None = None,
) -> FastAPI:
    """Build the FastAPI app, wiring in live references to the subsystems."""

    app = FastAPI(
        title="PMAI Core",
        version=__version__,
        description="PROMETEO Multimodal AI – monitoring and health API",
    )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/cameras")
    async def cameras() -> dict[str, Any]:
        if camera_manager is None:
            return {"cameras": []}
        infos = camera_manager.cameras
        return {
            "count": len(infos),
            "cameras": [
                {
                    "camera_id": info.camera_id,
                    "device_path": info.device_path,
                    "name": info.name,
                    "resolution": list(info.resolution),
                    "fps": info.fps,
                    "status": info.status.value,
                }
                for info in infos.values()
            ],
        }

    @app.get("/objects")
    async def objects() -> dict[str, Any]:
        if pipeline_engine is None:
            return {"identities": 0, "ids": []}
        registry = pipeline_engine.registry
        return {
            "identities": registry.size,
            "ids": registry.identity_ids,
        }

    @app.get("/trackers")
    async def trackers() -> dict[str, Any]:
        if pipeline_engine is None:
            return {"trackers": {}}
        return {
            "active_cameras": list(pipeline_engine.trackers.keys()),
        }

    @app.get("/cameras/{camera_id}/view", response_class=Response)
    async def camera_view(camera_id: str) -> Response:
        """Return the latest annotated frame (YOLO + ReID) as JPEG."""
        if pipeline_engine is None:
            raise HTTPException(status_code=503, detail="Pipeline not available")
        data = pipeline_engine.get_last_annotated(camera_id)
        if data is None:
            raise HTTPException(
                status_code=404,
                detail=f"No annotated frame for camera {camera_id}",
            )
        frame, tracked = data
        annotated = draw_detections(frame, tracked)
        _, jpeg = cv2.imencode(".jpg", annotated)
        return Response(
            content=jpeg.tobytes(),
            media_type="image/jpeg",
        )

    @app.get("/view", response_class=HTMLResponse)
    async def view_page() -> HTMLResponse:
        """HTML page that shows the first available camera view with auto-refresh."""
        if camera_manager is None or not camera_manager.cameras:
            return HTMLResponse(
                "<html><body><p>No cameras available.</p></body></html>",
                status_code=200,
            )
        first_cam_id = next(iter(camera_manager.cameras))
        html = f"""<!DOCTYPE html>
            <html>
            <head><title>PMAI View</title></head>
            <body>
            <h1>Camera: {first_cam_id}</h1>
            <img id="img" src="/cameras/{first_cam_id}/view" alt="Camera view" />
            <script>
                setInterval(function() {{
                document.getElementById("img").src = "/cameras/{first_cam_id}/view?t=" + Date.now();
                }}, 2000);
            </script>
            </body>
            </html>"""

        return HTMLResponse(html)

    return app
