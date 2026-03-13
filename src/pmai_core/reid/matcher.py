"""Cosine-similarity matcher for cross-camera Re-ID."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pmai_core.domain.tracked_object import TrackedObject
from pmai_core.reid.registry import GlobalRegistry


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

    def match(self, objects: list[TrackedObject]) -> list[TrackedObject]:
        """Assign or update ``global_id`` on each object based on embedding similarity."""
        for obj in objects:
            if obj.embedding is None:
                continue

            best_id, best_score = self._registry.query(obj.embedding)

            if best_id is not None and best_score >= self._threshold:
                obj.global_id = best_id
                self._registry.update(best_id, obj.embedding)
            else:
                new_id = self._registry.register(obj.embedding, obj.label)
                obj.global_id = new_id

        return objects

    @staticmethod
    def cosine_similarity(
        a: NDArray[np.float32],
        b: NDArray[np.float32],
    ) -> float:
        """Compute cosine similarity between two L2-normalised vectors."""
        return float(np.dot(a, b))
