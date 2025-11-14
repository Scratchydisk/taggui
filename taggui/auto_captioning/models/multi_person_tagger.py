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


logger = logging.getLogger(__name__)


class MultiPersonTagger(AutoCaptioningModel):
    """
    Multi-person tagger using YOLOv8 detection and WD Tagger.

    Detects people, tags each individually, extracts scene tags, and formats
    output as: person1: tags, person2: tags, scene: tags
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
        self.include_scene_tags = caption_settings.get('include_scene_tags', True)
        self.max_scene_tags = caption_settings.get('max_scene_tags', 20)
        self.max_tags_per_person = caption_settings.get('max_tags_per_person', 50)

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
        self.person_detector = PersonDetector(
            model_size=self.yolo_model_size,
            device=str(self.device),
            conf_threshold=self.detection_confidence,
            min_size=self.detection_min_size,
            max_people=self.detection_max_people
        )
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

    def generate_caption(
        self,
        model_inputs: PilImage.Image,
        image_prompt: str
    ) -> tuple[str, str]:
        """
        Generate multi-person caption.

        Pipeline:
        1. Detect people with YOLOv8
        2. If no people: fall back to standard WD Tagger on full image
        3. For each person: crop, tag with WD Tagger
        4. Tag full image and extract scene tags
        5. Format output as "person1: tags, person2: tags, scene: tags"

        Args:
            model_inputs: PIL Image
            image_prompt: Prompt (unused)

        Returns:
            Tuple of (caption, console_output_caption)
        """
        pil_image = model_inputs

        # Save image temporarily for YOLO detection
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            temp_path = tmp_file.name
            pil_image.save(temp_path)

        try:
            # Step 1: Detect people
            detections = self.person_detector.detect_people(temp_path)

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

            # Step 3: Tag each detected person
            person_tags_list = []
            for i, detection in enumerate(detections):
                try:
                    # Crop person
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

            # Step 5: Format output
            caption = self._format_output(person_tags_list, scene_tags)
            return caption, caption

        finally:
            # Clean up temporary file
            import os
            try:
                os.unlink(temp_path)
            except Exception as e:
                logger.warning(f"Could not delete temp file {temp_path}: {e}")

    def _format_output(
        self,
        person_tags_list: list[list[str]],
        scene_tags: list[str]
    ) -> str:
        """
        Format output as structured text.

        Args:
            person_tags_list: List of tag lists, one per person
            scene_tags: List of scene tags

        Returns:
            Formatted string: "person1: tags, person2: tags, scene: tags"
        """
        parts = []

        # Add person tags
        for i, tags in enumerate(person_tags_list):
            if tags:  # Only include if person has tags
                person_str = f"person{i+1}: {self.thread.tag_separator.join(tags)}"
                parts.append(person_str)

        # Add scene tags
        if scene_tags:
            scene_str = f"scene: {self.thread.tag_separator.join(scene_tags)}"
            parts.append(scene_str)

        # Join all parts with newline
        caption = '\n'.join(parts)
        return caption
