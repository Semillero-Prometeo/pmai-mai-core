"""Application bootstrap – wires all subsystems and starts the main loop."""

from __future__ import annotations

import asyncio
import signal

import structlog
import uvicorn

from pmai_core.api.router import create_api
from pmai_core.resources.camera.manager import CameraManager
from pmai_core.messaging.client import NATSClient
from pmai_core.resources.identification.service import IdentificationService
from pmai_core.settings import Settings
from pmai_core.pipeline.engine import PipelineEngine

logger = structlog.get_logger(__name__)


async def run_app(settings: Settings | None = None) -> None:
    """Entry point that bootstraps and runs the full application."""

    if settings is None:
        settings = Settings.from_toml()

    # _configure_logging(settings.app.log_level)
    logger.info("starting_pmai_core", app=settings.app.name)

    # --- Camera manager ---
    camera_manager = CameraManager(settings)
    discovered = camera_manager.start()
    logger.info("cameras_ready", count=len(discovered))

    # --- NATS ---
    nats_client = NATSClient(settings.nats)
    await nats_client.connect()

    # --- Identification Service (single instance for API and pipeline) ---
    identification_service = IdentificationService(settings, nats_client)

    # --- Pipeline Engine (uses same identification service) ---
    engine = PipelineEngine(
        camera_manager=camera_manager,
        identification_service=identification_service,
    )

    # --- FastAPI (uses same identification service) ---
    api = create_api(camera_manager=camera_manager, identification_service=identification_service)
    api_config = uvicorn.Config(
        api,
        host=settings.api.host,
        port=settings.api.port,
        log_level="warning",
    )
    api_server = uvicorn.Server(api_config)

    # --- Graceful shutdown ---
    shutdown_event = asyncio.Event()

    def _signal_handler() -> None:
        logger.info("shutdown_signal_received")
        shutdown_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

    # --- Run all tasks concurrently ---
    tasks = [
        asyncio.create_task(engine.run(), name="pipeline"),
        asyncio.create_task(api_server.serve(), name="api"),
    ]

    if settings.camera.auto_discover and settings.camera.source_mode == "v4l2":
        tasks.append(asyncio.create_task(camera_manager.run_polling_loop(), name="cam_poll"))

    # Wait for shutdown signal
    await shutdown_event.wait()

    # Cleanup
    logger.info("shutting_down")
    engine.stop()
    camera_manager.stop()
    await nats_client.close()

    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    logger.info("shutdown_complete")
