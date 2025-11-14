"""
Scene tag extraction utility.

Extracts scene/environment/setting tags from WD Tagger results by filtering
based on scene-related keywords.
"""

import logging
from typing import List, Tuple, Set


logger = logging.getLogger(__name__)


# Default scene-related keywords (using spaces, not underscores)
DEFAULT_SCENE_KEYWORDS = [
    # Locations
    'indoors', 'outdoors', 'building', 'room', 'street', 'city', 'urban',
    'countryside', 'rural', 'town',

    # Environments
    'forest', 'beach', 'mountain', 'field', 'desert', 'ocean', 'sea',
    'lake', 'river', 'water', 'park', 'garden', 'nature', 'landscape',

    # Settings/Rooms
    'classroom', 'bedroom', 'kitchen', 'bathroom', 'office', 'library',
    'shop', 'store', 'restaurant', 'cafe', 'bar', 'club', 'gym',
    'hospital', 'hotel', 'lobby', 'corridor', 'hallway', 'entrance',
    'living room', 'dining room', 'studio', 'workshop', 'garage',

    # Time/Weather
    'night', 'day', 'evening', 'morning', 'afternoon', 'sunset', 'sunrise',
    'dawn', 'dusk', 'cloudy', 'rain', 'raining', 'snow', 'snowing',
    'sunny', 'overcast', 'foggy', 'stormy',

    # Objects/Furniture (scene elements)
    'furniture', 'table', 'chair', 'desk', 'bed', 'sofa', 'couch',
    'bookshelf', 'shelf', 'cabinet', 'counter', 'cupboard',

    # Background elements
    'wall', 'window', 'door', 'floor', 'ceiling', 'roof', 'ground',
    'grass', 'tree', 'plant', 'flower', 'bush', 'fence', 'gate',

    # Lighting/Atmosphere
    'light', 'lighting', 'lamp', 'chandelier', 'candle', 'fire', 'fireplace',
    'shadow', 'silhouette', 'backlight', 'sunlight', 'moonlight',

    # Scene descriptors
    'background', 'scenery', 'landscape', 'panorama', 'view', 'scene',
    'setting', 'environment', 'location', 'place', 'area', 'space',

    # Additional common scene tags
    'bridge', 'path', 'road', 'sidewalk', 'pavement', 'stairs', 'steps',
    'bench', 'sign', 'pole', 'pillar', 'column', 'arch', 'dome',
    'tile', 'brick', 'stone', 'wood', 'wooden', 'metal', 'concrete',
    'carpet', 'rug', 'curtain', 'blinds', 'mirror', 'picture', 'painting',
    'poster', 'decoration', 'ornament',
]


class SceneExtractor:
    """
    Extracts scene-related tags from WD Tagger results.

    Filters tags based on configurable scene keywords to separate scene/
    environment tags from character/person tags.
    """

    def __init__(self, scene_keywords: List[str] = None):
        """
        Initialise the scene extractor.

        Args:
            scene_keywords: List of keywords to identify scene tags.
                          Uses DEFAULT_SCENE_KEYWORDS if None.
                          Keywords should use spaces, not underscores
                          (e.g., "wooden floor" not "wooden_floor").
        """
        if scene_keywords is None:
            scene_keywords = DEFAULT_SCENE_KEYWORDS

        # Convert to set for fast lookup and normalise to lowercase
        self.scene_keywords: Set[str] = {kw.lower() for kw in scene_keywords}

        logger.info(f"SceneExtractor initialised with {len(self.scene_keywords)} keywords")

    def extract_scene_tags(
        self,
        all_tags: List[Tuple[str, float]],
        max_tags: int = 20
    ) -> List[str]:
        """
        Extract scene-related tags from full WD Tagger results.

        Args:
            all_tags: List of (tag, confidence) tuples from WD Tagger.
                     Tags should already have underscores replaced with spaces.
            max_tags: Maximum number of scene tags to return

        Returns:
            List of scene tag names (without confidences), sorted by original
            confidence order, limited to max_tags.
        """
        scene_tags = []

        for tag, confidence in all_tags:
            # Normalise tag to lowercase for comparison
            tag_lower = tag.lower()

            # Check if tag contains any scene keywords
            is_scene_tag = False

            # Direct match or substring match
            for keyword in self.scene_keywords:
                if keyword == tag_lower or keyword in tag_lower:
                    is_scene_tag = True
                    break

            # Additional heuristics for scene tags
            if not is_scene_tag:
                # Tags starting with "background"
                if tag_lower.startswith('background'):
                    is_scene_tag = True
                # Tags ending with "bg"
                elif tag_lower.endswith(' bg') or tag_lower == 'bg':
                    is_scene_tag = True
                # Tags containing "background"
                elif 'background' in tag_lower:
                    is_scene_tag = True

            if is_scene_tag:
                scene_tags.append(tag)  # Keep original case

                # Stop if we've reached max_tags
                if len(scene_tags) >= max_tags:
                    break

        logger.debug(
            f"Extracted {len(scene_tags)} scene tags from "
            f"{len(all_tags)} total tags"
        )

        return scene_tags

    def add_keywords(self, keywords: List[str]) -> None:
        """
        Add additional scene keywords to the extractor.

        Args:
            keywords: List of keywords to add (will be normalised to lowercase)
        """
        new_keywords = {kw.lower() for kw in keywords}
        added = new_keywords - self.scene_keywords
        self.scene_keywords.update(new_keywords)

        if added:
            logger.info(f"Added {len(added)} new scene keywords")

    def remove_keywords(self, keywords: List[str]) -> None:
        """
        Remove scene keywords from the extractor.

        Args:
            keywords: List of keywords to remove
        """
        keywords_lower = {kw.lower() for kw in keywords}
        removed = self.scene_keywords & keywords_lower
        self.scene_keywords -= keywords_lower

        if removed:
            logger.info(f"Removed {len(removed)} scene keywords")

    def get_keywords(self) -> List[str]:
        """
        Get current list of scene keywords.

        Returns:
            Sorted list of scene keywords
        """
        return sorted(self.scene_keywords)

    def is_scene_tag(self, tag: str) -> bool:
        """
        Check if a single tag is considered a scene tag.

        Args:
            tag: Tag to check (case-insensitive)

        Returns:
            True if tag matches scene keywords, False otherwise
        """
        tag_lower = tag.lower()

        # Check keywords
        for keyword in self.scene_keywords:
            if keyword == tag_lower or keyword in tag_lower:
                return True

        # Check additional heuristics
        if (tag_lower.startswith('background') or
            tag_lower.endswith(' bg') or
            tag_lower == 'bg' or
            'background' in tag_lower):
            return True

        return False
