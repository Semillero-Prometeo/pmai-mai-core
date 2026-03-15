"""Cosine-similarity matcher for cross-camera Re-ID."""

from __future__ import annotations

import numpy as np
import structlog
from numpy.typing import NDArray

from pmai_core.domain.tracked_object import TrackedObject
from pmai_core.resources.reid.registry import GlobalRegistry

logger = structlog.get_logger(__name__)


class CosineMatcher:
    """Compare embeddings against the global identity gallery.

    When a tracked object's embedding exceeds ``similarity_threshold``
    against an existing global identity, it is associated with that
    identity.  Otherwise, a new global ID is created.
    """

    def __init__(
        self,
        registry: GlobalRegistry,
        similarity_threshold: float = 0.7,
    ) -> None:
        self._registry = registry
        self._threshold = similarity_threshold

    def match(
        self,
        objects: list[TrackedObject],
        *,
        camera_id: str = "",
    ) -> list[TrackedObject]:
        """Assign or update ``global_id`` on each object based on embedding similarity."""
        for obj in objects:
            if obj.embedding is None:
                continue

            best_id, best_score = self._registry.query(obj.embedding)

            if best_id is not None and best_score >= self._threshold:
                obj.global_id = best_id
                self._registry.update(
                    best_id, obj.embedding, camera_id=camera_id,
                )
                # logger.debug(
                #     "reid_matched_existing",
                #     track_id=obj.id,
                #     global_id=best_id,
                #     score=round(best_score, 3),
                #     camera_id=camera_id,
                # )
            else:
                new_id = self._registry.register(
                    obj.embedding, obj.label, camera_id=camera_id,
                )
                obj.global_id = new_id
                # logger.debug(
                #     "reid_new_identity",
                #     track_id=obj.id,
                #     global_id=new_id,
                #     best_existing_score=round(best_score, 3) if best_id else None,
                #     camera_id=camera_id,
                # )

        return objects

    @staticmethod
    def cosine_similarity(
        a: NDArray[np.float32],
        b: NDArray[np.float32],
    ) -> float:
        """Compute cosine similarity between two L2-normalised vectors."""
        return float(np.dot(a, b))
