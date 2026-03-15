"""Embedding extraction for Re-ID using an ONNX model (e.g. OSNet)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import structlog
from numpy.typing import NDArray

from pmai_core.domain.detection import BBox
from pmai_core.settings import ReIDSettings

logger = structlog.get_logger(__name__)

_REID_INPUT_H = 256
_REID_INPUT_W = 128


class EmbeddingExtractor:
    """Extract appearance embeddings from cropped detections using ONNX Runtime.

    Expected model: OSNet-ain-x0.25 (or compatible) with input shape
    ``(1, 3, 256, 128)`` and output shape ``(1, 512)``.
    """

    def __init__(self, settings: ReIDSettings) -> None:
        model_path = Path(settings.model_path)
        if not model_path.exists():
            logger.warning(
                "reid_model_not_found",
                path=str(model_path),
                msg="Embedding extraction will be disabled",
            )
            self._session: ort.InferenceSession | None = None
            return

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.intra_op_num_threads = 1
        sess_opts.inter_op_num_threads = 1

        self._session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name
        input_shape = self._session.get_inputs()[0].shape
        self._batch_size: int = int(input_shape[0]) if isinstance(input_shape[0], int) else 1

        self._warmup()
        logger.info(
            "reid_extractor_loaded",
            model=str(model_path),
            batch_size=self._batch_size,
        )

    def _warmup(self) -> None:
        """Run a single dummy inference to initialise ONNX Runtime thread pools.

        This MUST happen before any PyTorch inference to avoid an OpenMP
        thread-pool deadlock between the two runtimes.
        """
        if self._session is None:
            return
        input_shape = self._session.get_inputs()[0].shape
        shape = tuple(d if isinstance(d, int) else 1 for d in input_shape)
        dummy = np.zeros(shape, dtype=np.float32)
        self._session.run(None, {self._input_name: dummy})
        logger.debug("reid_warmup_complete", input_shape=list(input_shape))

    @property
    def is_available(self) -> bool:
        return self._session is not None

    def extract(
        self,
        frame: NDArray[np.uint8],
        bbox: BBox,
    ) -> NDArray[np.float32] | None:
        """Crop the bounding box from ``frame`` and return a normalised embedding."""
        if self._session is None:
            return None

        crop = self._crop_and_preprocess(frame, bbox)

        if self._batch_size > 1:
            padded = np.zeros(
                (self._batch_size, *crop.shape[1:]), dtype=np.float32,
            )
            padded[0] = crop[0]
            outputs = self._session.run(None, {self._input_name: padded})
            embedding: NDArray[np.float32] = outputs[0][0].flatten().astype(np.float32)
        else:
            outputs = self._session.run(None, {self._input_name: crop})
            embedding = outputs[0].flatten().astype(np.float32)

        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding

    def extract_batch(
        self,
        frame: NDArray[np.uint8],
        bboxes: list[BBox],
    ) -> list[NDArray[np.float32] | None]:
        """Extract embeddings for multiple bounding boxes."""
        return [self.extract(frame, bb) for bb in bboxes]

    @staticmethod
    def _crop_and_preprocess(
        frame: NDArray[np.uint8],
        bbox: BBox,
    ) -> NDArray[np.float32]:
        h, w = frame.shape[:2]
        x1 = max(0, bbox.xmin)
        y1 = max(0, bbox.ymin)
        x2 = min(w, bbox.xmax)
        y2 = min(h, bbox.ymax)
        crop = frame[y1:y2, x1:x2]

        crop = cv2.resize(crop, (_REID_INPUT_W, _REID_INPUT_H))
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        blob = crop.astype(np.float32) / 255.0

        # ImageNet-style normalisation
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        blob = (blob - mean) / std

        blob = np.transpose(blob, (2, 0, 1))
        return np.expand_dims(blob, axis=0)
