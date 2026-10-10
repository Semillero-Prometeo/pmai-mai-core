"""Startup warmup for Milvus, BLIP, embeddings, and Gemma."""

from __future__ import annotations

import structlog

logger = structlog.get_logger(__name__)


def prepare_runtime() -> None:
    try:
        from utils.milvus.conexion import get_client

        get_client()
        logger.info("runtime_milvus_ready")
    except Exception:
        logger.exception("runtime_milvus_prepare_failed")

    try:
        from model_context.model import import_Blip_model

        import_Blip_model()
        logger.info("runtime_blip_ready")
    except Exception:
        logger.exception("runtime_blip_prepare_failed")

    try:
        import utils.encoder.encoder  # noqa: F401

        logger.info("runtime_encoder_ready")
    except Exception:
        logger.exception("runtime_encoder_prepare_failed")

    try:
        from utils.gguf.ggufapi import precargar

        precargar()
        logger.info("runtime_gguf_ready")
    except Exception:
        logger.exception("runtime_gguf_prepare_failed")
