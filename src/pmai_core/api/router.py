"""FastAPI router with health, camera, and object monitoring endpoints."""

from __future__ import annotations

from typing import Any

import cv2
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response

from pmai_core import __version__
from pmai_core.resources.camera.manager import CameraManager
from pmai_core.pipeline.engine import PipelineEngine
from pmai_core.pipeline.global_objects import compute_global_objects_for_context
from pmai_core.resources.vision.overlay import draw_detections


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

    @app.get("/objects/tracked")
    async def objects_tracked() -> JSONResponse:
        """Return current global objects (one per id_global), deduplicated.

        Schema: id_global, etiqueta, confianza, contexto, sensores, cameras_seen, etc.
        This is the structured output that the LLM contextualizer will consume.
        """
        if pipeline_engine is None:
            return JSONResponse({"count": 0, "objects": []})
        annotated = pipeline_engine.all_last_annotated
        registry = pipeline_engine.registry
        global_objects = compute_global_objects_for_context(annotated, registry)
        return JSONResponse(
            {
                "count": len(global_objects),
                "objects": [o.model_dump() for o in global_objects],
            }
        )

    @app.get("/reid/status")
    async def reid_status() -> JSONResponse:
        """Summary of cross-camera ReID state: identities and which cameras see them."""
        if pipeline_engine is None:
            return JSONResponse({"error": "pipeline not available"}, status_code=503)
        registry = pipeline_engine.registry
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
        return JSONResponse(
            {
                "total_identities": len(identities),
                "cross_camera_identities": cross_count,
                "identities": identities,
            }
        )

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
        cameras_seen_map = _build_cameras_seen_map(pipeline_engine, tracked)
        annotated = draw_detections(
            frame,
            tracked,
            cameras_seen_map=cameras_seen_map,
        )
        _, jpeg = cv2.imencode(".jpg", annotated)
        return Response(
            content=jpeg.tobytes(),
            media_type="image/jpeg",
        )

    @app.get("/view", response_class=HTMLResponse)
    async def view_page() -> HTMLResponse:
        """HTML page that shows ALL cameras side by side with auto-refresh."""
        if camera_manager is None or not camera_manager.cameras:
            return HTMLResponse(
                "<html><body><p>No cameras available.</p></body></html>",
                status_code=200,
            )
        cam_ids = list(camera_manager.cameras.keys())
        html = _build_multicam_html(cam_ids)
        return HTMLResponse(html)

    return app


def _build_cameras_seen_map(
    engine: PipelineEngine,
    tracked: list[Any],
) -> dict[str, list[str]]:
    """Build {global_id: [cameras...]} for objects in the given tracked list."""
    result: dict[str, list[str]] = {}
    for obj in tracked:
        gid = obj.global_id
        if gid and gid not in result:
            result[gid] = engine.registry.get_cameras_for_identity(gid)
    return result


def _build_multicam_html(cam_ids: list[str]) -> str:
    n = len(cam_ids)
    cam_imgs = "\n".join(
        f"""
        <div class="cam-cell">
            <div class="cam-header">{cid}</div>
            <img id="img-{cid}" src="/cameras/{cid}/view" alt="{cid}" />
        </div>"""
        for cid in cam_ids
    )
    cam_refresh_js = "\n".join(
        f'document.getElementById("img-{cid}").src = "/cameras/{cid}/view?t=" + t;'
        for cid in cam_ids
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>PMAI – Multi-Camera ReID View</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
        font-family: system-ui, -apple-system, sans-serif;
        background: #111;
        color: #eee;
        padding: 12px;
    }}
    h1 {{
        text-align: center;
        margin-bottom: 8px;
        font-size: 1.3rem;
        color: #0ff;
    }}
    #reid-banner {{
        text-align: center;
        padding: 6px;
        margin-bottom: 10px;
        background: #1a1a2e;
        border-radius: 6px;
        font-size: 0.9rem;
    }}
    .grid {{
        display: grid;
        grid-template-columns: repeat({min(n, 3)}, 1fr);
        gap: 10px;
    }}
    .cam-cell {{
        background: #1a1a1a;
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #333;
    }}
    .cam-header {{
        padding: 6px 10px;
        background: #222;
        font-weight: 600;
        font-size: 0.85rem;
        color: #0ff;
    }}
    .cam-cell img {{
        width: 100%;
        display: block;
    }}
</style>
</head>
<body>
    <h1>PROMETEO – Multi-Camera ReID</h1>
    <div id="reid-banner">Loading ReID status...</div>
    <div class="grid">
        {cam_imgs}
    </div>
    <script>
        function refreshFrames() {{
            var t = Date.now();
            {cam_refresh_js}
        }}
        function refreshReidStatus() {{
            fetch("/reid/status")
                .then(r => r.json())
                .then(data => {{
                    var el = document.getElementById("reid-banner");
                    var cross = data.cross_camera_identities || 0;
                    var total = data.total_identities || 0;
                    var parts = ["Identities: " + total, "Cross-camera: " + cross];
                    if (data.identities) {{
                        data.identities.forEach(function(id) {{
                            if (id.cross_camera) {{
                                parts.push(
                                    '<span style="color:#0f0">' +
                                    id.global_id.replace("gid_", "") +
                                    " \\u2192 " + id.cameras_seen.join(", ") +
                                    '</span>'
                                );
                            }}
                        }});
                    }}
                    el.innerHTML = parts.join(" &nbsp;|&nbsp; ");
                }})
                .catch(function() {{}});
        }}
        setInterval(refreshFrames, 1000);
        setInterval(refreshReidStatus, 2000);
        refreshFrames();
        refreshReidStatus();
    </script>
</body>
</html>"""
