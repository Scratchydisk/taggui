"""
Utility components for auto-captioning models.

This package contains reusable utilities for detection, extraction, and
processing tasks used by auto-captioning models.
"""

from auto_captioning.utils.person_detector import PersonDetector
from auto_captioning.utils.scene_extractor import SceneExtractor

__all__ = ['PersonDetector', 'SceneExtractor']
