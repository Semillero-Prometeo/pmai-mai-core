"""Registers NATS request–reply subjects for PMAI Core (same pattern as ms-robotics)."""

from __future__ import annotations

from nats.aio.client import Client as NATSClient

from pmai_core.api.camera.camera_controller import CameraController
from pmai_core.api.health.health_controller import HealthController
from pmai_core.api.monitoring.monitoring_controller import MonitoringController
from pmai_core.api.vision.vision_controller import VisionController
from pmai_core.core.nats.interfaces.nats_interface import NatsSubscriber
from pmai_core.resources.camera.manager import CameraManager
from pmai_core.resources.identification.service import IdentificationService
from pmai_core.settings import Settings


def create_subscribers(
    nats_client: NATSClient,
    settings: Settings,
    camera_manager: CameraManager,
    identification_service: IdentificationService,
) -> list[NatsSubscriber]:
    _ = nats_client
    ms = settings.nats.ms_name

    health_controller = HealthController(settings)
    camera_controller = CameraController(camera_manager, identification_service)
    monitoring_controller = MonitoringController(identification_service)
    vision_controller = VisionController(camera_manager, identification_service)

    return [
        NatsSubscriber(
            controller=health_controller.health,
            subject=f"{ms}.healthService.health",
        ),
        NatsSubscriber(
            controller=camera_controller.get_cameras,
            subject=f"{ms}.cameraService.getCameras",
        ),
        NatsSubscriber(
            controller=camera_controller.get_camera_view,
            subject=f"{ms}.cameraService.getCameraView",
        ),
        NatsSubscriber(
            controller=monitoring_controller.get_objects,
            subject=f"{ms}.monitoringService.getObjects",
        ),
        NatsSubscriber(
            controller=monitoring_controller.get_trackers,
            subject=f"{ms}.monitoringService.getTrackers",
        ),
        NatsSubscriber(
            controller=monitoring_controller.get_tracked_objects,
            subject=f"{ms}.monitoringService.getTrackedObjects",
        ),
        NatsSubscriber(
            controller=monitoring_controller.get_reid_status,
            subject=f"{ms}.monitoringService.getReidStatus",
        ),
        NatsSubscriber(
            controller=vision_controller.get_snapshot,
            subject=f"{ms}.visionService.getSnapshot",
        ),
    ]
