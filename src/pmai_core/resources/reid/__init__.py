"""Cross-camera Re-Identification subsystem."""

from pmai_core.reid.extractor import EmbeddingExtractor
from pmai_core.reid.matcher import CosineMatcher
from pmai_core.reid.registry import GlobalRegistry

__all__ = ["CosineMatcher", "EmbeddingExtractor", "GlobalRegistry"]
