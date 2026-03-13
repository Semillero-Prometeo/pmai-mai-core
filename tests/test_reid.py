"""Tests for Re-ID matching and global registry."""

from __future__ import annotations

import numpy as np

from pmai_core.domain.detection import BBox
from pmai_core.domain.tracked_object import TrackedObject
from pmai_core.reid.matcher import CosineMatcher
from pmai_core.reid.registry import GlobalRegistry


class TestGlobalRegistry:
    def test_register_and_query(self) -> None:
        registry = GlobalRegistry(max_size=100)
        emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        gid = registry.register(emb, label="person")
        assert registry.size == 1

        best_id, score = registry.query(emb)
        assert best_id == gid
        assert score > 0.99

    def test_query_empty_registry(self) -> None:
        registry = GlobalRegistry()
        emb = np.array([1.0, 0.0], dtype=np.float32)
        best_id, score = registry.query(emb)
        assert best_id is None
        assert score == 0.0

    def test_update_running_average(self) -> None:
        registry = GlobalRegistry()
        emb1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        gid = registry.register(emb1, label="person")

        emb2 = np.array([0.9, 0.1, 0.0], dtype=np.float32)
        emb2 = emb2 / np.linalg.norm(emb2)
        registry.update(gid, emb2)

        _, score = registry.query(emb1)
        assert score > 0.9  # averaged embedding should still be close

    def test_eviction_on_max_size(self) -> None:
        registry = GlobalRegistry(max_size=2)
        registry.register(np.array([1.0, 0.0], dtype=np.float32), label="a")
        registry.register(np.array([0.0, 1.0], dtype=np.float32), label="b")
        registry.register(np.array([1.0, 1.0], dtype=np.float32), label="c")
        assert registry.size == 2

    def test_cameras_tracked(self) -> None:
        registry = GlobalRegistry()
        emb = np.array([1.0, 0.0], dtype=np.float32)
        gid = registry.register(emb, label="person", camera_id="cam_0")
        registry.update(gid, emb, camera_id="cam_1")
        cameras = registry.get_cameras_for_identity(gid)
        assert set(cameras) == {"cam_0", "cam_1"}


class TestCosineMatcher:
    def test_assigns_global_id_to_known_identity(self) -> None:
        registry = GlobalRegistry()
        emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        gid = registry.register(emb, label="person")

        matcher = CosineMatcher(registry, similarity_threshold=0.7)
        obj = TrackedObject(
            id=1,
            label="person",
            confidence=0.9,
            bbox=BBox(xmin=0, ymin=0, xmax=100, ymax=100),
            camera_id="cam_0",
            embedding=emb,
        )
        result = matcher.match([obj])
        assert result[0].global_id == gid

    def test_creates_new_identity_when_no_match(self) -> None:
        registry = GlobalRegistry()
        emb_existing = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        registry.register(emb_existing, label="person")

        matcher = CosineMatcher(registry, similarity_threshold=0.99)
        emb_new = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        obj = TrackedObject(
            id=2,
            label="person",
            confidence=0.8,
            bbox=BBox(xmin=10, ymin=10, xmax=110, ymax=110),
            camera_id="cam_1",
            embedding=emb_new,
        )
        result = matcher.match([obj])
        assert result[0].global_id != ""
        assert registry.size == 2

    def test_cosine_similarity_identical(self) -> None:
        v = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        v = v / np.linalg.norm(v)
        assert abs(CosineMatcher.cosine_similarity(v, v) - 1.0) < 1e-6

    def test_cosine_similarity_orthogonal(self) -> None:
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        assert abs(CosineMatcher.cosine_similarity(a, b)) < 1e-6
