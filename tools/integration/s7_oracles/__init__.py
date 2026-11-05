"""Actual-input pixel oracles used by the formal S7 reviewer."""
from .review import review_device, review_offscreen, review_surface

__all__ = ["review_device", "review_offscreen", "review_surface"]
