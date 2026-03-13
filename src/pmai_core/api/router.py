"""FastAPI router with health, camera, and object monitoring endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI

from pmai_core import __version__


def create_api(
    camera_manager: Any = None,
    pipeline_engine: Any = None,
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

    return app
