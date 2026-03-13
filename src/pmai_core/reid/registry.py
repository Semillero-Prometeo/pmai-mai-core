"""In-memory global identity registry for cross-camera Re-ID."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

import numpy as np
import structlog
from numpy.typing import NDArray

logger = structlog.get_logger(__name__)


@dataclass
class _Identity:
    global_id: str
    label: str
    embedding: NDArray[np.float32]
    update_count: int = 1
    cameras_seen: set[str] = field(default_factory=set)


class GlobalRegistry:
    """Maintains a gallery of known identities keyed by global ID.

    Embeddings are stored as running averages so the representation improves
    over time as the same person is observed from multiple viewpoints.
    """

    def __init__(self, max_size: int = 1000) -> None:
        self._identities: dict[str, _Identity] = {}
        self._max_size = max_size

    @property
    def size(self) -> int:
        return len(self._identities)

    @property
    def identity_ids(self) -> list[str]:
        return list(self._identities.keys())

    def register(
        self,
        embedding: NDArray[np.float32],
        label: str,
        camera_id: str = "",
    ) -> str:
        """Create a new global identity and return its ID."""
        if len(self._identities) >= self._max_size:
            self._evict_oldest()

        global_id = f"gid_{uuid.uuid4().hex[:8]}"
        self._identities[global_id] = _Identity(
            global_id=global_id,
            label=label,
            embedding=embedding.copy(),
            cameras_seen={camera_id} if camera_id else set(),
        )
        logger.info("identity_registered", global_id=global_id, label=label)
        return global_id

    def update(
        self,
        global_id: str,
        embedding: NDArray[np.float32],
        camera_id: str = "",
    ) -> None:
        """Update the running-average embedding for an existing identity."""
        identity = self._identities.get(global_id)
        if identity is None:
            return

        n = identity.update_count
        identity.embedding = (identity.embedding * n + embedding) / (n + 1)
        # Re-normalise
        norm = np.linalg.norm(identity.embedding)
        if norm > 0:
            identity.embedding = identity.embedding / norm
        identity.update_count = n + 1

        if camera_id:
            identity.cameras_seen.add(camera_id)

    def query(
        self,
        embedding: NDArray[np.float32],
    ) -> tuple[str | None, float]:
        """Find the best-matching identity. Returns ``(global_id, score)``."""
        if not self._identities:
            return None, 0.0

        best_id: str | None = None
        best_score = -1.0

        for gid, identity in self._identities.items():
            score = float(np.dot(embedding, identity.embedding))
            if score > best_score:
                best_score = score
                best_id = gid

        return best_id, best_score

    def get_cameras_for_identity(self, global_id: str) -> list[str]:
        identity = self._identities.get(global_id)
        return list(identity.cameras_seen) if identity else []

    def _evict_oldest(self) -> None:
        """Remove the identity with the lowest update count (least observed)."""
        if not self._identities:
            return
        worst_id = min(self._identities, key=lambda k: self._identities[k].update_count)
        del self._identities[worst_id]
        logger.debug("identity_evicted", global_id=worst_id)
