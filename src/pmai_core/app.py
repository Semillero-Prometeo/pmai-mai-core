"""Application bootstrap – wires subsystems, NATS RPC, and the pipeline loop."""

from __future__ import annotations

import asyncio
import signal

import structlog

from pmai_core.core.nats.nats_client import nats_handler
from pmai_core.core.nats.nats_subscribers import create_subscribers
from pmai_core.messaging.client import NATSClient
from pmai_core.pipeline.engine import PipelineEngine
from pmai_core.resources.audio.service import AudioService
from pmai_core.resources.camera.manager import CameraManager
from pmai_core.resources.identification.service import IdentificationService
from pmai_core.settings import Settings

logger = structlog.get_logger(__name__)


async def run_app(settings: Settings | None = None) -> None:
    """Entry point that bootstraps NATS RPC, the pipeline, and graceful shutdown."""

    if settings is None:
        settings = Settings.from_toml()

    logger.info("starting_pmai_core", app=settings.app.name)

    camera_manager = CameraManager(settings)
    discovered = camera_manager.start()
    logger.info("cameras_ready", count=len(discovered))

    nats_client = NATSClient(settings.nats)
    await nats_client.connect()

    identification_service = IdentificationService(settings, nats_client)
    audio_service = AudioService(settings)
    audio_service.bind_rpc(nats_handler.nc, asyncio.get_running_loop())
    audio_service.start()

    engine = PipelineEngine(
        camera_manager=camera_manager,
        identification_service=identification_service,
    )

    await nats_handler.connect(settings.nats.url)

    subscribers = create_subscribers(
        nats_handler.nc,
        settings=settings,
        camera_manager=camera_manager,
        identification_service=identification_service,
        audio_service=audio_service,
    )
    for subscriber in subscribers:
        await nats_handler.subscribe(subscriber)

    shutdown_event = asyncio.Event()

    def _signal_handler() -> None:
        logger.info("shutdown_signal_received")
        shutdown_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

    tasks = [
        asyncio.create_task(engine.run(), name="pipeline"),
        asyncio.create_task(audio_service.run_polling_loop(), name="audio_poll"),
    ]

    if settings.camera.auto_discover and settings.camera.source_mode == "v4l2":
        tasks.append(asyncio.create_task(camera_manager.run_polling_loop(), name="cam_poll"))

    await shutdown_event.wait()

    logger.info("shutting_down")
    engine.stop()
    audio_service.stop()
    camera_manager.stop()
    await nats_handler.disconnect()
    await nats_client.close()

    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    logger.info("shutdown_complete")
