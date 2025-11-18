"""
Multi-person tagger model.

Detects multiple people in images and generates per-person tags plus scene tags
using YOLOv8 for detection and WD Tagger for tagging.
"""

import logging
from datetime import datetime

import numpy as np
from PIL import Image as PilImage

import auto_captioning.captioning_thread as captioning_thread
from auto_captioning.auto_captioning_model import AutoCaptioningModel
from auto_captioning.models.wd_tagger import WdTaggerModel
from auto_captioning.utils.person_detector import PersonDetector
from auto_captioning.utils.scene_extractor import SceneExtractor
from utils.image import Image
from utils.settings import DEFAULT_SETTINGS, get_settings


logger = logging.getLogger(__name__)


class MultiPersonTagger(AutoCaptioningModel):
    """
    Multi-person tagger using YOLOv8 detection and WD Tagger.

    Detects people, tags each individually, extracts scene tags, and formats
    output as: person1, tags, person2, tags, scene, tags
    """

    image_mode = 'RGBA'  # Same as WD Tagger

    def __init__(
        self,
        captioning_thread_: 'captioning_thread.CaptioningThread',
        caption_settings: dict
    ):
        super().__init__(captioning_thread_, caption_settings)

        # Extract multi-person specific settings
        self.detection_confidence = caption_settings.get('detection_confidence', 0.5)
        self.detection_min_size = caption_settings.get('detection_min_size', 50)
        self.detection_max_people = caption_settings.get('detection_max_people', 10)
        self.crop_padding = caption_settings.get('crop_padding', 10)
        self.yolo_model_size = caption_settings.get('yolo_model_size', 'm')
        self.split_merged_people = caption_settings.get('split_merged_people', True)
        self.mask_overlapping_people = caption_settings.get('mask_overlapping_people', True)
        self.masking_method = caption_settings.get('masking_method', 'Bounding box')
        self.preserve_target_bbox = caption_settings.get('preserve_target_bbox', True)
        self.include_scene_tags = caption_settings.get('include_scene_tags', True)
        self.max_scene_tags = caption_settings.get('max_scene_tags', 20)
        self.max_tags_per_person = caption_settings.get('max_tags_per_person', 50)

        # Experimental mask refinement parameters
        self.mask_erosion_size = caption_settings.get('mask_erosion_size', 0)
        self.mask_dilation_size = caption_settings.get('mask_dilation_size', 0)
        self.mask_blur_size = caption_settings.get('mask_blur_size', 0)

        # Parse person aliases (comma-separated)
        person_aliases_str = caption_settings.get('person_aliases', '').strip()
        if person_aliases_str:
            # Split by comma and strip whitespace from each alias
            self.person_aliases = [alias.strip() for alias in person_aliases_str.split(',') if alias.strip()]
        else:
            self.person_aliases = []

        # Caption mode: 'lora_tags' or 'fine_tune_caption'
        self.caption_mode = caption_settings.get('caption_mode', 'lora_tags')
        self.description_model_name = caption_settings.get('description_model', 'vikhyatk/moondream2')

        # WD Tagger settings (construct from mp_ prefixed settings)
        self.wd_tagger_settings = {
            'show_probabilities': False,  # Not shown in console for multi-person
            'min_probability': caption_settings.get('mp_wd_tagger_min_probability', 0.35),
            'max_tags': self.max_tags_per_person,  # Use max_tags_per_person
            'tags_to_exclude': caption_settings.get('mp_wd_tagger_tags_to_exclude', '')
        }
        self.wd_model_id = caption_settings.get('wd_model', 'SmilingWolf/wd-eva02-large-tagger-v3')

        # Components (initialized in get_model)
        self.person_detector = None
        self.wd_model = None
        self.scene_extractor = None
        self.description_model = None  # Lazy-loaded VLM for fine-tune captions

    def get_error_message(self) -> str | None:
        """Validate settings and return error message if invalid."""
        if self.detection_confidence < 0.1 or self.detection_confidence > 1.0:
            return 'Detection confidence must be between 0.1 and 1.0'
        if self.detection_max_people < 1:
            return 'Maximum people must be at least 1'
        return None

    def load_processor_and_model(self):
        """
        Override parent to always initialize components.

        Multi-person tagger has component instances that must be initialized
        for each model instance, even if model caching is enabled.
        """
        # Always initialize components (don't use parent's caching)
        logger.info("Initializing multi-person tagger components...")
        self.processor = self.get_processor()
        self.model = self.get_model()

    @staticmethod
    def parse_person_count_from_tags(tags: tuple[str, ...]) -> int:
        """
        Parse expected person count from WD Tagger tags.

        Looks for tags like '2boys', '3girls', '1boy', '1girl', 'solo', etc.

        Args:
            tags: Tuple of tag strings from WD Tagger

        Returns:
            Expected number of people (0 if cannot determine)
        """
        import re

        person_count = 0

        # Patterns to match
        # Match Nboys, Ngirls (e.g., "2boys", "3girls", "6+boys")
        for tag in tags:
            # Match patterns like "2boys", "3girls", "4+boys"
            match = re.match(r'(\d+)\+?(boy|girl)s?', tag)
            if match:
                count = int(match.group(1))
                person_count += count
                continue

            # Match "1boy", "1girl"
            if tag in ('1boy', '1girl'):
                person_count += 1
                continue

            # Match "solo" = 1 person
            if tag == 'solo':
                person_count = max(person_count, 1)
                continue

        return person_count

    @staticmethod
    def split_detection_by_connected_components(detection: dict) -> list[dict]:
        """
        Split a single detection into multiple detections based on connected components.

        Uses segmentation mask to find separate connected blobs and creates
        individual detections with bounding boxes for each.

        Args:
            detection: Detection dict with 'mask', 'bbox', 'confidence', etc.

        Returns:
            List of detection dicts (one per connected component).
            If no mask or only 1 component, returns original detection in a list.
        """
        import cv2

        mask = detection.get('mask')
        if mask is None:
            # No mask available, cannot split
            logger.debug("Cannot split detection: no segmentation mask")
            return [detection]

        # Convert boolean mask to uint8 for OpenCV
        mask_uint8 = (mask.astype(np.uint8) * 255)

        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask_uint8, connectivity=8
        )

        # Label 0 is background, so actual components start at 1
        num_components = num_labels - 1

        if num_components <= 1:
            # Only one blob, no need to split
            logger.debug(f"Detection has only {num_components} component(s), not splitting")
            return [detection]

        logger.info(f"Splitting detection into {num_components} components")

        split_detections = []

        for i in range(1, num_labels):  # Skip background (0)
            # Create mask for this component
            component_mask = (labels == i)

            # Get bounding box from stats
            # stats format: [left, top, width, height, area]
            x, y, w, h, area = stats[i]

            # Skip very small components (noise)
            if area < 100:  # Minimum 100 pixels
                logger.debug(f"Skipping component {i}: area {area} too small")
                continue

            # Create new detection for this component
            new_detection = {
                'bbox': [x, y, x + w, y + h],
                'confidence': detection['confidence'],  # Keep original confidence
                'area': w * h,  # Bbox area
                'center_y': y + h // 2,
                'mask': component_mask
            }

            split_detections.append(new_detection)
            logger.debug(f"Created component {i}: bbox=({x},{y},{x+w},{y+h}), area={area}")

        return split_detections if split_detections else [detection]

    def get_processor(self):
        """No processor needed for WD Tagger-based models."""
        return None

    def get_model(self):
        """
        Load all model components.

        Returns dict with person_detector, wd_model, and scene_extractor.
        """
        logger.info("Loading multi-person tagger components...")

        # Initialize PersonDetector for YOLO
        # Use segmentation model if:
        # 1. Masking method is 'Segmentation', OR
        # 2. Split merged people is enabled (requires segmentation masks)
        use_segmentation = (
            (self.mask_overlapping_people and self.masking_method == 'Segmentation') or
            self.split_merged_people
        )
        self.person_detector = PersonDetector(
            model_size=self.yolo_model_size,
            device=str(self.device),
            conf_threshold=self.detection_confidence,
            min_size=self.detection_min_size,
            max_people=self.detection_max_people,
            use_segmentation=use_segmentation
        )
        if use_segmentation:
            logger.info("PersonDetector loaded with segmentation")
        else:
            logger.info("PersonDetector loaded")

        # Initialize WdTaggerModel (reuse existing implementation)
        # Handle models_directory_path like WdTagger does
        wd_model_id = self.wd_model_id
        models_directory_path = self.thread.models_directory_path
        if models_directory_path:
            tags_path = (models_directory_path / wd_model_id / 'selected_tags.csv')
            if tags_path.is_file():
                wd_model_id = str(models_directory_path / wd_model_id)

        self.wd_model = WdTaggerModel(wd_model_id)
        logger.info(f"WdTaggerModel loaded: {wd_model_id}")

        # Initialize SceneExtractor
        self.scene_extractor = SceneExtractor()
        logger.info("SceneExtractor loaded")

        # Return dict with all components
        return {
            'person_detector': self.person_detector,
            'wd_model': self.wd_model,
            'scene_extractor': self.scene_extractor
        }

    def get_captioning_message(
        self,
        are_multiple_images_selected: bool,
        captioning_start_datetime: datetime
    ) -> str:
        """Get status message shown during captioning."""
        if are_multiple_images_selected:
            captioning_start_datetime_string = (
                self.get_captioning_start_datetime_string(
                    captioning_start_datetime))
            return (f'Generating multi-person tags... (start time: '
                    f'{captioning_start_datetime_string})')
        return 'Generating multi-person tags...'

    def get_model_inputs(self, image_prompt: str, image: Image) -> PilImage.Image:
        """
        Load and preprocess image.

        For multi-person tagger, we just load the PIL image since we'll
        handle person detection and cropping in generate_caption.

        Args:
            image_prompt: Prompt (unused for this model)
            image: Image object

        Returns:
            PIL Image in RGBA mode
        """
        pil_image = self.load_image(image)
        return pil_image

    def _load_edited_masks(self, image_path, detections: list) -> bool:
        """
        Load edited masks, enabled states, and aliases from sidecar .masks.npz file if it exists.

        Args:
            image_path: Path to the image file
            detections: List of detection dictionaries to update with edited data

        Returns:
            True if data was loaded, False otherwise
        """
        from pathlib import Path

        # Check for sidecar file
        mask_file_path = Path(image_path).with_suffix(Path(image_path).suffix + '.masks.npz')

        if not mask_file_path.exists():
            return False

        try:
            # Load mask data (allow_pickle for alias strings)
            mask_data = np.load(mask_file_path, allow_pickle=True)

            # Determine how many people are saved (look for person_N_bbox keys)
            saved_person_count = 0
            for key in mask_data.keys():
                if key.startswith('person_') and key.endswith('_bbox'):
                    # Extract person index
                    person_num = int(key.split('_')[1])
                    saved_person_count = max(saved_person_count, person_num + 1)

            logger.debug(f"Found {saved_person_count} saved people, currently have {len(detections)} detections")

            # Apply masks, enabled states, and aliases to existing detections
            data_loaded = 0
            for i, detection in enumerate(detections):
                person_key = f'person_{i}_mask'
                bbox_key = f'person_{i}_bbox'
                enabled_key = f'person_{i}_enabled'
                alias_key = f'person_{i}_alias'

                # Load mask if present
                if person_key in mask_data and bbox_key in mask_data:
                    # Verify bbox matches (detections should be consistent)
                    saved_bbox = mask_data[bbox_key]
                    current_bbox = np.array(detection['bbox'])

                    # Allow small differences due to detection variance
                    if np.allclose(saved_bbox, current_bbox, atol=10):
                        detection['mask'] = mask_data[person_key]
                        data_loaded += 1
                        logger.debug(f"Loaded edited mask for person {i+1}")
                    else:
                        logger.warning(
                            f"Bbox mismatch for person {i+1}, skipping edited mask. "
                            f"Saved: {saved_bbox}, Current: {current_bbox}"
                        )

                # Load enabled state if present
                if enabled_key in mask_data:
                    enabled_value = mask_data[enabled_key][0]
                    detection['enabled'] = bool(enabled_value)
                    logger.debug(f"Loaded enabled state for person {i+1}: {detection['enabled']}")

                # Load alias if present
                if alias_key in mask_data:
                    alias_value = mask_data[alias_key][0]
                    detection['alias'] = str(alias_value) if alias_value else ''
                    logger.debug(f"Loaded alias for person {i+1}: '{detection['alias']}'")

            # Load any additional manually-added people (inverse crops, split by line, etc.)
            if saved_person_count > len(detections):
                logger.info(
                    f"Found {saved_person_count - len(detections)} manually-added people, "
                    f"loading them..."
                )
                for i in range(len(detections), saved_person_count):
                    person_key = f'person_{i}_mask'
                    bbox_key = f'person_{i}_bbox'
                    enabled_key = f'person_{i}_enabled'
                    alias_key = f'person_{i}_alias'

                    # Check if this person has required data
                    if person_key not in mask_data or bbox_key not in mask_data:
                        logger.warning(f"Skipping person {i+1}: missing mask or bbox data")
                        continue

                    # Reconstruct detection from saved data
                    saved_bbox = mask_data[bbox_key]
                    saved_mask = mask_data[person_key]

                    # Calculate detection properties from bbox
                    x1, y1, x2, y2 = saved_bbox
                    bbox_width = x2 - x1
                    bbox_height = y2 - y1
                    area = bbox_width * bbox_height
                    center_y = (y1 + y2) // 2

                    # Load enabled state (default True)
                    enabled = True
                    if enabled_key in mask_data:
                        enabled_value = mask_data[enabled_key][0]
                        enabled = bool(enabled_value)

                    # Load alias (default empty)
                    alias = ''
                    if alias_key in mask_data:
                        alias_value = mask_data[alias_key][0]
                        alias = str(alias_value) if alias_value else ''

                    # Create new detection
                    new_detection = {
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': 1.0,  # Manually added, so confidence is 1.0
                        'area': int(area),
                        'center_y': int(center_y),
                        'mask': saved_mask,
                        'enabled': enabled,
                        'alias': alias
                    }

                    detections.append(new_detection)
                    data_loaded += 1
                    logger.info(
                        f"Loaded manually-added person {i+1} "
                        f"(alias: '{alias}', enabled: {enabled}, bbox: {saved_bbox})"
                    )

            if data_loaded > 0:
                logger.info(f"Loaded detection data from {mask_file_path.name}")
                return True
            else:
                logger.warning(f"No matching detection data found in {mask_file_path.name}")
                return False

        except Exception as e:
            logger.error(f"Failed to load detection data from {mask_file_path}: {e}")
            return False

    def _refine_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply morphological operations and blur to refine mask.

        Args:
            mask: Boolean mask array (HxW)

        Returns:
            Refined boolean mask
        """
        if self.mask_erosion_size == 0 and self.mask_dilation_size == 0 and self.mask_blur_size == 0:
            # No refinement needed
            return mask

        import cv2

        # Convert to uint8 for OpenCV operations
        mask_uint8 = (mask.astype(np.uint8) * 255)

        # Apply erosion (shrink mask)
        if self.mask_erosion_size > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.mask_erosion_size, self.mask_erosion_size))
            mask_uint8 = cv2.erode(mask_uint8, kernel, iterations=1)
            logger.debug(f"Applied mask erosion: {self.mask_erosion_size}px")

        # Apply dilation (expand mask)
        if self.mask_dilation_size > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.mask_dilation_size, self.mask_dilation_size))
            mask_uint8 = cv2.dilate(mask_uint8, kernel, iterations=1)
            logger.debug(f"Applied mask dilation: {self.mask_dilation_size}px")

        # Apply Gaussian blur
        if self.mask_blur_size > 0:
            # Ensure kernel size is odd
            kernel_size = self.mask_blur_size if self.mask_blur_size % 2 == 1 else self.mask_blur_size + 1
            mask_uint8 = cv2.GaussianBlur(mask_uint8, (kernel_size, kernel_size), 0)
            logger.debug(f"Applied mask blur: {kernel_size}px kernel")

        # Convert back to boolean
        refined_mask = mask_uint8 > 127

        return refined_mask

    def _preprocess_image_for_wd_tagger(self, pil_image: PilImage.Image) -> np.ndarray:
        """
        Preprocess PIL image for WD Tagger (same as wd_tagger.py).

        Args:
            pil_image: PIL Image in RGBA mode

        Returns:
            Numpy array ready for WD Tagger inference
        """
        # Add white background for transparent areas
        canvas = PilImage.new('RGBA', pil_image.size, (255, 255, 255))
        canvas.alpha_composite(pil_image)
        pil_image = canvas.convert('RGB')

        # Pad to square
        max_dimension = max(pil_image.size)
        canvas = PilImage.new('RGB', (max_dimension, max_dimension), (255, 255, 255))
        horizontal_padding = (max_dimension - pil_image.width) // 2
        vertical_padding = (max_dimension - pil_image.height) // 2
        canvas.paste(pil_image, (horizontal_padding, vertical_padding))

        # Resize to model input dimensions
        _, input_dimension, *_ = (self.wd_model.inference_session.get_inputs()[0].shape)
        if max_dimension != input_dimension:
            input_dimensions = (input_dimension, input_dimension)
            canvas = canvas.resize(input_dimensions, resample=PilImage.Resampling.BICUBIC)

        # Convert to numpy array
        image_array = np.array(canvas, dtype=np.float32)

        # Reverse color channels (RGB -> BGR)
        image_array = image_array[:, :, ::-1]

        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)

        return image_array

    def _extract_segmented_person(
        self,
        image: PilImage.Image,
        detection: dict,
        padding: int = 10
    ) -> PilImage.Image:
        """
        Extract only the segmented person pixels on a white background.

        This is the cleanest approach for overlapping people - we extract ONLY
        the pixels that belong to the target person (using their segmentation mask)
        and place them on a white background. No masking of other people needed.

        Args:
            image: Original PIL Image
            detection: Detection dict with 'mask' and 'bbox' keys
            padding: Padding around the person in pixels

        Returns:
            PIL Image with only the target person on white background
        """
        import numpy as np

        mask = detection.get('mask')
        if mask is None:
            # Fall back to regular bbox crop if no mask
            logger.warning("No segmentation mask available, falling back to bbox crop")
            return self.person_detector.crop_person(image, detection['bbox'], padding)

        # Get image dimensions
        width, height = image.size

        # Convert image to numpy array
        img_array = np.array(image)

        # Create white background (same size as image)
        num_channels = img_array.shape[2] if len(img_array.shape) == 3 else 1
        white_value = [255] * num_channels if num_channels > 1 else 255
        result_array = np.full_like(img_array, white_value)

        # Copy only the pixels where the mask is True
        result_array[mask] = img_array[mask]

        # Convert back to PIL Image
        result_image = PilImage.fromarray(result_array)

        # Find the bounding box of the mask (to crop tightly)
        # Get mask coordinates where mask is True
        mask_coords = np.argwhere(mask)
        if len(mask_coords) == 0:
            # Empty mask - fall back to bbox
            logger.warning("Empty segmentation mask, falling back to bbox crop")
            return self.person_detector.crop_person(image, detection['bbox'], padding)

        # mask_coords is in (y, x) format
        y_coords = mask_coords[:, 0]
        x_coords = mask_coords[:, 1]

        # Get tight bounds
        y_min, y_max = y_coords.min(), y_coords.max()
        x_min, x_max = x_coords.min(), x_coords.max()

        # Add padding
        crop_x1 = max(0, x_min - padding)
        crop_y1 = max(0, y_min - padding)
        crop_x2 = min(width, x_max + padding + 1)  # +1 because max is inclusive
        crop_y2 = min(height, y_max + padding + 1)

        # Crop to the padded bounds
        cropped = result_image.crop((crop_x1, crop_y1, crop_x2, crop_y2))

        logger.debug(
            f"Extracted segmented person: mask_bounds=({x_min},{y_min},{x_max},{y_max}), "
            f"crop=({crop_x1},{crop_y1},{crop_x2},{crop_y2})"
        )

        return cropped

    def _mask_overlapping_bboxes(
        self,
        image: PilImage.Image,
        target_bbox: list[int],
        all_detections: list[dict],
        target_index: int,
        padding: int
    ) -> PilImage.Image:
        """
        Create a copy of the image with other people masked out (bounding box method).

        Args:
            image: Original PIL Image
            target_bbox: Bounding box of the person we want to keep [x1, y1, x2, y2]
            all_detections: List of all person detections
            target_index: Index of the target person in all_detections
            padding: Crop padding to determine the crop region

        Returns:
            PIL Image with other people masked to neutral grey
        """
        from PIL import ImageDraw

        # Create a copy to mask
        masked_image = image.copy()
        draw = ImageDraw.Draw(masked_image)

        # Calculate the crop region that will be extracted
        width, height = image.size
        tx1, ty1, tx2, ty2 = target_bbox
        crop_x1 = max(0, tx1 - padding)
        crop_y1 = max(0, ty1 - padding)
        crop_x2 = min(width, tx2 + padding)
        crop_y2 = min(height, ty2 + padding)

        # Mask other people whose bboxes overlap with the crop region
        for i, detection in enumerate(all_detections):
            if i == target_index:
                continue  # Don't mask the target person

            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox

            # Check if this bbox overlaps with the crop region
            if not (x2 < crop_x1 or x1 > crop_x2 or y2 < crop_y1 or y1 > crop_y2):
                # Overlaps - mask only the overlapping part with neutral grey (127, 127, 127)
                # Calculate the intersection of the bbox with the crop region
                overlap_x1 = max(x1, crop_x1)
                overlap_y1 = max(y1, crop_y1)
                overlap_x2 = min(x2, crop_x2)
                overlap_y2 = min(y2, crop_y2)
                draw.rectangle([overlap_x1, overlap_y1, overlap_x2, overlap_y2], fill=(127, 127, 127))
                logger.debug(f"Masked person {i+1} bbox for person {target_index+1} crop")

        return masked_image

    def _mask_overlapping_segmentation(
        self,
        image: PilImage.Image,
        target_bbox: list[int],
        all_detections: list[dict],
        target_index: int,
        padding: int
    ) -> PilImage.Image:
        """
        Create a copy of the image with other people masked out (segmentation method).

        Args:
            image: Original PIL Image
            target_bbox: Bounding box of the person we want to keep [x1, y1, x2, y2]
            all_detections: List of all person detections (must include 'mask' key)
            target_index: Index of the target person in all_detections
            padding: Crop padding to determine the crop region

        Returns:
            PIL Image with other people masked to neutral grey using segmentation masks
        """
        import numpy as np
        from PIL import ImageDraw

        # Create a copy to mask
        masked_image = image.copy()
        img_array = np.array(masked_image)

        # Calculate the crop region that will be extracted
        width, height = image.size
        tx1, ty1, tx2, ty2 = target_bbox
        crop_x1 = max(0, tx1 - padding)
        crop_y1 = max(0, ty1 - padding)
        crop_x2 = min(width, tx2 + padding)
        crop_y2 = min(height, ty2 + padding)

        # Determine number of channels (RGB=3, RGBA=4)
        num_channels = img_array.shape[2] if len(img_array.shape) == 3 else 1
        grey_value = [127] * num_channels if num_channels > 1 else 127

        # If preserve_target_bbox is True, create a protected region for the target's full bbox
        protected_region = None
        if self.preserve_target_bbox:
            protected_region = np.zeros((height, width), dtype=bool)
            # Protect the target person's full bounding box area
            protected_region[ty1:ty2, tx1:tx2] = True
            logger.debug(f"Created protected bbox region for person {target_index+1}: ({tx1},{ty1})-({tx2},{ty2})")

        # Mask other people using their segmentation masks
        for i, detection in enumerate(all_detections):
            if i == target_index:
                continue  # Don't mask the target person

            mask = detection.get('mask')
            if mask is None:
                # Fall back to bounding box if mask not available
                bbox = detection['bbox']
                x1, y1, x2, y2 = bbox
                if not (x2 < crop_x1 or x1 > crop_x2 or y2 < crop_y1 or y1 > crop_y2):
                    # Mask the overlapping part of the bbox
                    overlap_x1 = max(x1, crop_x1)
                    overlap_y1 = max(y1, crop_y1)
                    overlap_x2 = min(x2, crop_x2)
                    overlap_y2 = min(y2, crop_y2)

                    # If protecting target bbox, don't mask protected pixels
                    if protected_region is not None:
                        bbox_mask = np.zeros((height, width), dtype=bool)
                        bbox_mask[overlap_y1:overlap_y2, overlap_x1:overlap_x2] = True
                        # Only mask pixels outside the protected region
                        pixels_to_mask = bbox_mask & ~protected_region
                        img_array[pixels_to_mask] = grey_value
                    else:
                        img_array[overlap_y1:overlap_y2, overlap_x1:overlap_x2] = grey_value

                    logger.debug(f"Masked person {i+1} bbox (no mask available) for person {target_index+1} crop")
                continue

            # Check if mask shape matches image shape
            if mask.shape[0] != height or mask.shape[1] != width:
                logger.warning(f"Mask shape {mask.shape} doesn't match image shape ({height}, {width})")
                continue

            # Apply mask refinement (erosion, dilation, blur)
            mask = self._refine_mask(mask)

            # Only mask the pixels that belong to this person AND overlap with the crop region
            # Create a mask for the crop region
            crop_mask = np.zeros_like(mask, dtype=bool)
            crop_mask[crop_y1:crop_y2, crop_x1:crop_x2] = True

            # Only mask pixels that are both part of the person AND in the crop region
            pixels_to_mask = mask & crop_mask

            # If protecting target bbox, exclude protected pixels
            if protected_region is not None:
                pixels_to_mask = pixels_to_mask & ~protected_region

            if pixels_to_mask.any():
                # Mask only the overlapping pixels
                img_array[pixels_to_mask] = grey_value
                logger.debug(f"Masked person {i+1} segmentation for person {target_index+1} crop")

        # Convert back to PIL Image
        masked_image = PilImage.fromarray(img_array)
        return masked_image

    def generate_caption(
        self,
        model_inputs: PilImage.Image,
        image_prompt: str,
        image=None
    ) -> tuple[str, str]:
        """
        Generate multi-person caption.

        Pipeline:
        1. Detect people with YOLOv8
        2. If no people: fall back to standard WD Tagger on full image
        3. For each person: crop, tag with WD Tagger
        4. Tag full image and extract scene tags
        5. Format output as "person1, tags, person2, tags, scene, tags"

        Args:
            model_inputs: PIL Image
            image_prompt: Prompt (unused)
            image: Optional Image object with path attribute

        Returns:
            Tuple of (caption, console_output_caption)
        """
        pil_image = model_inputs

        # Save image temporarily for YOLO detection
        import tempfile
        import uuid
        from pathlib import Path

        # Read temp file location setting
        settings = get_settings()
        temp_file_location = settings.value(
            'temp_file_location',
            defaultValue=DEFAULT_SETTINGS['temp_file_location'],
            type=str
        )

        # Determine where to create temp file based on setting
        use_source_folder = (temp_file_location == 'Source folder' and
                            image and hasattr(image, 'path'))

        if use_source_folder:
            # Create temp file in same directory as the image
            temp_dir = image.path.parent
            temp_filename = f'.taggui_tmp_person_detection_{uuid.uuid4().hex[:8]}.png'
            temp_path = temp_dir / temp_filename
            pil_image.save(temp_path)
        else:
            # Use system temp directory
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                temp_path = tmp_file.name
                pil_image.save(temp_path)

        try:
            # Step 1: Detect people (with iterative detection if split_merged_people enabled)
            if self.split_merged_people:
                # Tag full image to get expected person count
                logger.info("Split merged people enabled: tagging full image to get expected count")
                image_array = self._preprocess_image_for_wd_tagger(pil_image)
                full_image_tags, _ = self.wd_model.generate_tags(
                    image_array,
                    self.wd_tagger_settings
                )

                expected_person_count = self.parse_person_count_from_tags(full_image_tags)
                logger.info(f"Expected people: {expected_person_count}")

                # Use iterative detection if we expect multiple people
                if expected_person_count > 0:
                    detections, initial_count = self.person_detector.detect_people_iteratively(
                        str(temp_path),
                        expected_person_count,
                        max_iterations=3
                    )
                    logger.info(
                        f"Iterative detection: found {len(detections)} people "
                        f"(initial: {initial_count}, expected: {expected_person_count})"
                    )
                else:
                    # No expected people, use standard detection
                    detections = self.person_detector.detect_people(str(temp_path))
            else:
                # Standard detection (no splitting)
                detections = self.person_detector.detect_people(str(temp_path))

            # Load edited masks from sidecar file if available
            if detections and image and hasattr(image, 'path'):
                self._load_edited_masks(image.path, detections)

            # Step 2: If no people detected, fall back to standard WD Tagger
            if not detections:
                logger.info("No people detected, using standard WD Tagger")
                image_array = self._preprocess_image_for_wd_tagger(pil_image)
                tags, probabilities = self.wd_model.generate_tags(
                    image_array,
                    self.wd_tagger_settings
                )
                caption = self.thread.tag_separator.join(tags)
                return caption, caption

            # Branch based on caption mode
            logger.info(f"Caption mode: {self.caption_mode}")
            if self.caption_mode == 'fine_tune_caption':
                # Fine-tune mode: Generate VLM descriptions
                logger.info("Entering fine-tune caption mode")
                caption = self._generate_fine_tune_caption(pil_image, detections)
                return caption, caption

            # LoRA mode: Continue with tag generation
            # Step 3: Tag each detected person (skip disabled ones)
            person_tags_list = []
            person_aliases = []  # Track aliases for enabled people
            enabled_person_index = 0  # Counter for enabled people only
            for i, detection in enumerate(detections):
                # Skip if person is disabled
                if not detection.get('enabled', True):
                    logger.info(f"Skipping person {i+1} (disabled)")
                    continue

                try:
                    # Store alias for this person (for output formatting)
                    # Use detection alias if set, otherwise fall back to control panel aliases
                    detection_alias = detection.get('alias', '').strip()
                    if detection_alias:
                        # Use alias from preview dialog
                        person_aliases.append(detection_alias)
                        logger.debug(f"Person {i+1}: using preview dialog alias '{detection_alias}'")
                    elif enabled_person_index < len(self.person_aliases):
                        # Use alias from control panel settings
                        control_panel_alias = self.person_aliases[enabled_person_index]
                        person_aliases.append(control_panel_alias)
                        logger.debug(f"Person {i+1}: using control panel alias '{control_panel_alias}'")
                    else:
                        # No alias available
                        person_aliases.append('')
                        logger.debug(f"Person {i+1}: no alias")

                    enabled_person_index += 1
                    # Extract person using segmentation or bbox+masking
                    if self.masking_method == 'Segmentation' and detection.get('mask') is not None:
                        # Extract ONLY the segmented person on white background
                        cropped = self._extract_segmented_person(
                            pil_image,
                            detection,
                            padding=self.crop_padding
                        )
                    elif self.mask_overlapping_people and len(detections) > 1:
                        # Use bounding box masking (mask out other people, then crop)
                        image_to_crop = self._mask_overlapping_bboxes(
                            pil_image,
                            detection['bbox'],
                            detections,
                            i,
                            self.crop_padding
                        )
                        # Crop person
                        cropped = self.person_detector.crop_person(
                            image_to_crop,
                            detection['bbox'],
                            padding=self.crop_padding
                        )
                    else:
                        # No masking - just crop the bounding box
                        cropped = self.person_detector.crop_person(
                            pil_image,
                            detection['bbox'],
                            padding=self.crop_padding
                        )

                    # Preprocess crop for WD Tagger
                    crop_array = self._preprocess_image_for_wd_tagger(cropped)

                    # Generate tags for this person
                    tags, probabilities = self.wd_model.generate_tags(
                        crop_array,
                        self.wd_tagger_settings
                    )

                    # Keep top N tags (without probabilities)
                    person_tags = list(tags[:self.max_tags_per_person])
                    person_tags_list.append(person_tags)

                    logger.debug(f"Person {i+1}: {len(person_tags)} tags")

                except Exception as e:
                    logger.error(f"Error tagging person {i+1}: {e}")
                    # Skip this person, continue with others
                    person_tags_list.append([])

            # Step 4: Tag full image and extract scene tags
            scene_tags = []
            if self.include_scene_tags:
                try:
                    image_array = self._preprocess_image_for_wd_tagger(pil_image)
                    all_tags, all_probabilities = self.wd_model.generate_tags(
                        image_array,
                        self.wd_tagger_settings
                    )

                    # Extract scene tags
                    tags_with_probs = list(zip(all_tags, all_probabilities))
                    scene_tags = self.scene_extractor.extract_scene_tags(
                        tags_with_probs,
                        max_tags=self.max_scene_tags
                    )

                    logger.debug(f"Scene: {len(scene_tags)} tags")

                except Exception as e:
                    logger.error(f"Error extracting scene tags: {e}")
                    scene_tags = []

            # Step 5: Format output (with person aliases)
            caption = self._format_output(person_tags_list, scene_tags, person_aliases)
            return caption, caption

        finally:
            # Clean up temporary file
            import os
            try:
                os.unlink(str(temp_path))
            except Exception as e:
                logger.warning(f"Could not delete temp file {temp_path}: {e}")

    def _format_output(
        self,
        person_tags_list: list[list[str]],
        scene_tags: list[str],
        person_aliases: list[str] = None
    ) -> str:
        """
        Format output as structured text.

        Args:
            person_tags_list: List of tag lists, one per person
            scene_tags: List of scene tags
            person_aliases: Optional list of aliases for each person

        Returns:
            Formatted string with person aliases or default personN labels
        """
        if person_aliases is None:
            person_aliases = []

        parts = []

        # Add person tags
        for i, tags in enumerate(person_tags_list):
            if tags:  # Only include if person has tags
                # Use alias if available and non-empty, otherwise fall back to personN
                if i < len(person_aliases) and person_aliases[i]:
                    person_label = person_aliases[i]
                else:
                    person_label = f"person{i+1}"

                person_str = f"{person_label}, {self.thread.tag_separator.join(tags)}"
                parts.append(person_str)

        # Add scene tags
        if scene_tags:
            scene_str = f"scene, {self.thread.tag_separator.join(scene_tags)}"
            parts.append(scene_str)

        # Join all parts with newline
        caption = '\n'.join(parts)
        return caption

    def _load_description_model(self):
        """Lazy-load the VLM model for natural language descriptions."""
        if self.description_model is not None:
            return

        try:
            # Import and instantiate the appropriate model
            from auto_captioning.models_list import get_model_class
            from transformers import AutoConfig

            logger.info(f"Loading description model: {self.description_model_name}")
            model_class = get_model_class(self.description_model_name)

            # Create a caption_settings dict for the VLM by copying current settings
            # and overriding the model_id
            vlm_settings = self.caption_settings.copy()
            vlm_settings['model_id'] = self.description_model_name
            vlm_settings['prompt'] = ''  # No prompt needed for descriptions

            # Disable flash attention for models that default to it (Phi-3, etc.)
            # Must modify config BEFORE loading model
            if 'phi-3' in self.description_model_name.lower():
                logger.info("Disabling flash attention for Phi-3...")
                config = AutoConfig.from_pretrained(
                    self.description_model_name,
                    trust_remote_code=True
                )
                config._attn_implementation = 'eager'
                vlm_settings['_model_config'] = config  # Pass modified config

            # Instantiate the VLM model
            self.description_model = model_class(self.thread, vlm_settings)

            # Load the model
            self.description_model.load_processor_and_model()

            logger.info(f"Description model loaded: {self.description_model_name}")

        except ImportError as e:
            error_msg = str(e)
            if 'einops' in error_msg:
                logger.error(f"Failed to load {self.description_model_name}: Missing 'einops' package. Install with: pip install einops")
                raise ImportError(f"Description model {self.description_model_name} requires 'einops' package. Install with: pip install einops") from e
            elif 'flash_attn' in error_msg or 'flash-attn' in error_msg:
                logger.error(f"Failed to load {self.description_model_name}: Missing 'flash-attn' package. Install with: pip install flash-attn")
                raise ImportError(f"Description model {self.description_model_name} requires 'flash-attn' package. Install with: pip install flash-attn") from e
            else:
                raise
        except Exception as e:
            logger.error(f"Failed to load description model {self.description_model_name}: {e}")
            raise

    @staticmethod
    def clean_vlm_output(description: str, prompt: str = '') -> str:
        """
        Clean up VLM output by removing prompt echo, extra whitespace, etc.

        Args:
            description: Raw VLM output
            prompt: The prompt that was used (to remove echoes)

        Returns:
            Cleaned description
        """
        if not description:
            return "A person in the image."

        # Remove prompt if echoed
        if prompt and description.startswith(prompt):
            description = description[len(prompt):].strip()

        # Remove common prefixes that VLMs add
        prefixes_to_remove = [
            "The person is ",
            "This person is ",
            "The image shows ",
            "In the image, ",
            "Description: ",
            "A: ",
            "Answer: ",
            "Sure, ",
            "Sure! ",
        ]
        for prefix in prefixes_to_remove:
            if description.lower().startswith(prefix.lower()):
                description = description[len(prefix):].strip()

        # Ensure starts with capital
        if description:
            description = description[0].upper() + description[1:]

        # Ensure ends with period
        if description and not description.endswith(('.', '!', '?')):
            description += '.'

        # Remove multiple spaces
        description = ' '.join(description.split())

        # Truncate if too long (max 50 words to be safe)
        words = description.split()
        if len(words) > 50:
            description = ' '.join(words[:50]) + '.'

        return description

    @staticmethod
    def detect_gender_from_tags(tags: list[str]) -> str:
        """
        Detect gender from WD tags.

        Args:
            tags: List of WD tags

        Returns:
            'female', 'male', or 'unknown'
        """
        if not tags:
            return 'unknown'

        female_indicators = ['girl', 'woman', 'female', 'feminine', 'lady', 'gal']
        male_indicators = ['boy', 'man', 'male', 'masculine', 'guy', 'gentleman']

        tags_lower = [tag.lower() for tag in tags]
        tags_str = ' '.join(tags_lower)

        # Check female indicators
        if any(indicator in tags_str for indicator in female_indicators):
            return 'female'

        # Check male indicators
        if any(indicator in tags_str for indicator in male_indicators):
            return 'male'

        return 'unknown'

    def describe_person_region(
        self,
        pil_image: PilImage.Image,
        detection: dict,
        person_index: int
    ) -> str:
        """
        Generate natural language description of a person region using VLM.

        Args:
            pil_image: Full PIL image
            detection: Detection dict with 'bbox' and optional 'mask'
            person_index: Index of person (for logging)

        Returns:
            Natural language description (15-35 words)
        """
        # Ensure description model is loaded
        self._load_description_model()

        # Create a temporary Image object for the VLM
        # VLMs expect an Image object with path and image attributes
        from utils.image import Image
        import tempfile
        from pathlib import Path
        import os

        # Create temp file
        tmp_fd, tmp_name = tempfile.mkstemp(suffix='.png')
        os.close(tmp_fd)  # Close the file descriptor
        tmp_path = Path(tmp_name)

        try:
            # Crop to person region
            bbox = detection['bbox']
            cropped = self.person_detector.crop_person(
                pil_image,
                bbox,
                padding=self.crop_padding
            )

            # Save cropped region to temp file
            cropped.save(tmp_path)

            # Create Image object wrapper with dimensions
            temp_image = Image(path=tmp_path, dimensions=cropped.size)
            temp_image.image = cropped  # Set the PIL image directly

            # Craft prompt for VLM
            base_prompt = (
                "Describe only what you can clearly see of this person in one sentence. "
                "Focus on visible clothing, appearance, and pose. "
                "Do not infer or imagine details that are not visible."
            )

            # Apply model-specific prompt formatting (e.g., Phi-3 needs special tokens)
            image_prompt = self.description_model.format_prompt(base_prompt)

            logger.debug(f"Calling VLM with model: {self.description_model_name}")
            logger.debug(f"Base prompt: {base_prompt}")
            logger.debug(f"Formatted prompt: {image_prompt}")
            logger.debug(f"Temp image path: {tmp_path}")
            logger.debug(f"Cropped image size: {cropped.size}")

            # Get model inputs (preprocessed by VLM's processor)
            # This handles model-specific formatting and preprocessing
            model_inputs = self.description_model.get_model_inputs(image_prompt, temp_image)
            logger.debug(f"Model inputs type: {type(model_inputs)}")
            logger.debug(f"Model inputs keys: {model_inputs.keys() if hasattr(model_inputs, 'keys') else 'N/A'}")

            # Generate description using VLM
            logger.debug("Calling generate_caption...")
            description, _ = self.description_model.generate_caption(
                model_inputs=model_inputs,
                image_prompt=image_prompt,
                image=temp_image
            )
            logger.debug(f"Raw VLM output: {description}")

            # Clean up description
            description = self.clean_vlm_output(description, image_prompt)

            logger.debug(f"Person {person_index + 1} description: {description}")
            return description

        except Exception as e:
            logger.error(f"Failed to generate description for person {person_index + 1}: {e}", exc_info=True)
            return f"[ERROR: Description generation failed - {type(e).__name__}]"

        finally:
            # Always clean up temp file
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception as cleanup_error:
                logger.warning(f"Failed to clean up temp file {tmp_path}: {cleanup_error}")

    def describe_scene(
        self,
        pil_image: PilImage.Image
    ) -> str:
        """
        Generate brief scene/setting description.

        Args:
            pil_image: Full PIL image

        Returns:
            Brief scene description (5-15 words) or empty string
        """
        self._load_description_model()

        # Create a temporary Image object for the VLM
        from utils.image import Image
        import tempfile
        from pathlib import Path
        import os

        # Create temp file
        tmp_fd, tmp_name = tempfile.mkstemp(suffix='.png')
        os.close(tmp_fd)  # Close the file descriptor
        tmp_path = Path(tmp_name)

        try:
            # Save image to temp file
            pil_image.save(tmp_path)

            # Create Image object wrapper with dimensions
            temp_image = Image(path=tmp_path, dimensions=pil_image.size)
            temp_image.image = pil_image

            # Craft prompt for VLM
            base_prompt = (
                "Describe only the visible setting and background in one brief sentence. "
                "Focus on what is actually visible: surfaces, objects, lighting. "
                "Do not describe people or infer details you cannot see."
            )

            # Apply model-specific prompt formatting (e.g., Phi-3 needs special tokens)
            image_prompt = self.description_model.format_prompt(base_prompt)

            # Get model inputs (preprocessed by VLM's processor)
            # This handles model-specific formatting and preprocessing
            model_inputs = self.description_model.get_model_inputs(image_prompt, temp_image)

            # Generate description using VLM
            logger.debug(f"Generating scene description with {self.description_model_name}...")
            description, _ = self.description_model.generate_caption(
                model_inputs=model_inputs,
                image_prompt=image_prompt,
                image=temp_image
            )
            logger.debug(f"Raw scene output: {description}")

            # Clean up description
            description = self.clean_vlm_output(description, image_prompt)
            logger.debug(f"Cleaned scene description: {description}")
            logger.debug(f"Word count: {len(description.split())}")

            # Only return if substantial description (filter out generic responses)
            if len(description.split()) >= 3:
                logger.debug(f"Scene description accepted: {description}")
                return description

            logger.debug(f"Scene description rejected (too short or empty)")
            return ""

        except Exception as e:
            logger.error(f"Failed to generate scene description: {e}", exc_info=True)
            return ""

        finally:
            # Always clean up temp file
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception as cleanup_error:
                logger.warning(f"Failed to clean up temp file {tmp_path}: {cleanup_error}")

    def _assign_fine_tune_aliases(
        self,
        num_people: int,
        person_tags_list: list[list[str]] = None
    ) -> list[str]:
        """
        Assign aliases for fine-tune mode.

        Priority:
        1. Use custom aliases from main UI (person_aliases setting)
        2. Fall back to person1, person2, person3, etc.

        Args:
            num_people: Number of detected people
            person_tags_list: Optional list of tags per person (for gender detection)

        Returns:
            List of aliases
        """
        aliases = []
        for i in range(num_people):
            # First check if custom alias is available from main UI
            if i < len(self.person_aliases) and self.person_aliases[i]:
                aliases.append(self.person_aliases[i])
            # Fall back to person1, person2, etc.
            else:
                aliases.append(f'person{i + 1}')

        return aliases

    def _format_fine_tune_output(
        self,
        person_descriptions: list[tuple[str, str]],
        scene_description: str
    ) -> str:
        """
        Format output for fine-tune mode.

        Args:
            person_descriptions: List of (alias, description) tuples
            scene_description: Optional scene description

        Returns:
            Multi-line formatted caption:
            sksA: [description]
            sksB: [description]
            Scene: [description]
        """
        lines = []

        # Add person descriptions
        for alias, description in person_descriptions:
            lines.append(f"{alias}: {description}")

        # Add scene description if present
        if scene_description:
            lines.append(f"Scene: {scene_description}")

        return '\n'.join(lines)

    def _generate_fine_tune_caption(
        self,
        pil_image: PilImage.Image,
        detections: list[dict]
    ) -> str:
        """
        Generate fine-tune style caption using VLM descriptions.

        Args:
            pil_image: PIL Image
            detections: List of detection dicts with 'bbox', 'mask', 'enabled', etc.

        Returns:
            Multi-line formatted caption:
            sksA: [description]
            sksB: [description]
            Scene: [description]
        """
        logger.info(f"Generating fine-tune caption for {len(detections)} detected people")

        # First pass: get WD tags quickly (only for gender detection if 3+ people)
        person_tags_list = []
        enabled_detections = [d for d in detections if d.get('enabled', True)]

        if len(enabled_detections) > 2:
            # Need gender detection for 3+ people
            for i, detection in enumerate(enabled_detections):
                try:
                    bbox = detection['bbox']
                    cropped = self.person_detector.crop_person(
                        pil_image,
                        bbox,
                        padding=self.crop_padding
                    )

                    # Quick WD tag for gender detection
                    crop_array = self._preprocess_image_for_wd_tagger(cropped)
                    tags, _ = self.wd_model.generate_tags(
                        crop_array,
                        self.wd_tagger_settings
                    )
                    person_tags_list.append(list(tags)[:20])  # Just need a few for gender

                except Exception as e:
                    logger.error(f"Error getting tags for person {i+1}: {e}")
                    person_tags_list.append([])

        # Assign aliases
        aliases = self._assign_fine_tune_aliases(
            num_people=len(enabled_detections),
            person_tags_list=person_tags_list if person_tags_list else None
        )

        # Generate VLM description for each person
        person_descriptions = []
        for i, detection in enumerate(enabled_detections):
            alias = aliases[i]

            # Generate natural language description
            description = self.describe_person_region(
                pil_image=pil_image,
                detection=detection,
                person_index=i
            )

            person_descriptions.append((alias, description))

        # Generate scene description
        scene_description = ""
        if self.include_scene_tags:
            scene_description = self.describe_scene(pil_image)

        # Format output
        caption = self._format_fine_tune_output(person_descriptions, scene_description)

        logger.info(f"Generated fine-tune caption with {len(person_descriptions)} people")
        return caption
