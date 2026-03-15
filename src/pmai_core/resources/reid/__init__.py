"""Cross-camera Re-Identification subsystem."""

from pmai_core.resources.reid.extractor import EmbeddingExtractor
from pmai_core.resources.reid.matcher import CosineMatcher
from pmai_core.resources.reid.registry import GlobalRegistry

__all__ = ["CosineMatcher", "EmbeddingExtractor", "GlobalRegistry"]
