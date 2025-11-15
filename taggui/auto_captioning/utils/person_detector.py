"""
Person detection utility using YOLOv8.

Provides person detection and cropping functionality for multi-person tagging.
"""

import logging
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO


logger = logging.getLogger(__name__)


class PersonDetector:
    """
    Wrapper for YOLOv8 person detection.

    Detects people in images, filters by confidence and size, and provides
    cropping functionality for detected bounding boxes.
    """

    def __init__(
        self,
        model_size: str = 'm',
        device: str = None,
        conf_threshold: float = 0.5,
        min_size: int = 50,
        max_people: int = 10,
        use_segmentation: bool = False
    ):
        """
        Initialise the person detector.

        Args:
            model_size: YOLO model size (n, s, m, l, x). Default: 'm' (medium)
            device: Device to run on ('cuda' or 'cpu'). Auto-detected if None.
            conf_threshold: Confidence threshold for detections (0.0-1.0)
            min_size: Minimum bounding box dimension (width or height) in pixels
            max_people: Maximum number of people to detect per image
            use_segmentation: Use YOLOv8-seg for instance segmentation. Default: False
        """
        # Device selection
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        # Detection parameters
        self.conf_threshold = conf_threshold
        self.min_size = min_size
        self.max_people = max_people
        self.use_segmentation = use_segmentation

        # Load YOLOv8 model
        model_suffix = '-seg' if use_segmentation else ''
        model_type = 'segmentation' if use_segmentation else 'detection'
        logger.info(f"Loading YOLOv8{model_size}{model_suffix} ({model_type}) on {device}...")
        try:
            model_name = f"yolov8{model_size}{model_suffix}.pt"
            self.model = YOLO(model_name)
            self.model.to(device)
            logger.info(f"YOLOv8{model_size}{model_suffix} loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load YOLOv8 model: {e}")
            raise RuntimeError(
                f"Could not load YOLOv8 model '{model_name}'. "
                f"The model will be auto-downloaded on first use if not found. "
                f"Error: {e}"
            ) from e

    def detect_people(
        self,
        image_path: str,
        conf_threshold: float = None
    ) -> List[Dict]:
        """
        Detect people in an image.

        Args:
            image_path: Path to image file
            conf_threshold: Override confidence threshold for this detection.
                          Uses instance threshold if None.

        Returns:
            List of detection dictionaries with format:
            [
                {
                    'bbox': [x1, y1, x2, y2],  # Bounding box coordinates
                    'confidence': float,        # Detection confidence
                    'area': int,                # Bounding box area
                    'center_y': int,            # Vertical center position
                    'mask': np.ndarray or None  # Segmentation mask (HxW bool array) if segmentation enabled
                },
                ...
            ]
            Sorted by area (largest first), then by center_y (top to bottom).
            Limited to max_people detections.
            Returns empty list if no people detected or on error.
        """
        if conf_threshold is None:
            conf_threshold = self.conf_threshold

        try:
            # Run YOLO detection (class 0 is 'person' in COCO dataset)
            results = self.model(
                image_path,
                conf=conf_threshold,
                classes=[0],
                verbose=False  # Suppress YOLO output
            )

            detections = []
            for result in results:
                boxes = result.boxes
                masks = result.masks if self.use_segmentation and hasattr(result, 'masks') and result.masks is not None else None

                # Get original image shape for mask resizing
                orig_shape = result.orig_shape if hasattr(result, 'orig_shape') else None

                if self.use_segmentation:
                    if masks is not None:
                        logger.debug(f"Segmentation enabled: Got {len(masks)} masks for {len(boxes)} boxes, orig_shape={orig_shape}")
                    else:
                        logger.warning("Segmentation enabled but no masks returned from YOLO model")

                for box_idx, box in enumerate(boxes):
                    # Extract bounding box coordinates
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = [int(coord) for coord in xyxy]

                    # Calculate dimensions
                    width = x2 - x1
                    height = y2 - y1

                    # Filter by minimum size
                    if width < self.min_size or height < self.min_size:
                        logger.debug(
                            f"Filtered detection: size {width}x{height} "
                            f"< min_size {self.min_size}"
                        )
                        continue

                    # Calculate area and center position for sorting
                    area = width * height
                    center_y = (y1 + y2) // 2

                    # Extract segmentation mask if available
                    mask = None
                    if masks is not None and box_idx < len(masks):
                        try:
                            # Get mask data as numpy array (HxW)
                            # For YOLOv8-seg, masks.data contains the mask for each detection
                            mask_data = masks.data[box_idx].cpu().numpy()

                            # Resize mask to original image dimensions
                            if orig_shape is not None and mask_data.shape != orig_shape[:2]:
                                # Convert to PIL Image for resizing
                                mask_img = Image.fromarray((mask_data * 255).astype(np.uint8))
                                # Resize to original image dimensions (height, width)
                                mask_img_resized = mask_img.resize((orig_shape[1], orig_shape[0]), Image.Resampling.BILINEAR)
                                # Convert back to numpy boolean array
                                mask = np.array(mask_img_resized) > 127
                                logger.debug(f"Resized mask from {mask_data.shape} to {mask.shape}")
                            else:
                                mask = mask_data > 0.5  # Convert to boolean mask

                            logger.debug(f"Extracted segmentation mask for person {box_idx+1}: shape={mask.shape}")
                        except Exception as e:
                            logger.warning(f"Failed to extract mask for detection {box_idx}: {e}")
                            mask = None

                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': float(box.conf[0]),
                        'area': area,
                        'center_y': center_y,
                        'mask': mask
                    })

            # Sort by area (largest first), then by Y position (top to bottom)
            detections.sort(key=lambda d: (-d['area'], d['center_y']))

            # Limit to max_people
            if len(detections) > self.max_people:
                logger.warning(
                    f"Detected {len(detections)} people, "
                    f"limiting to {self.max_people}"
                )
                detections = detections[:self.max_people]

            logger.info(f"Detected {len(detections)} person(s) in {image_path}")
            return detections

        except Exception as e:
            logger.error(f"Error detecting people in {image_path}: {e}")
            return []

    def crop_person(
        self,
        image: Image.Image,
        bbox: List[int],
        padding: int = 10
    ) -> Image.Image:
        """
        Crop a person from an image with optional padding.

        Args:
            image: PIL Image object
            bbox: Bounding box [x1, y1, x2, y2]
            padding: Padding around the bbox in pixels

        Returns:
            Cropped PIL Image

        Raises:
            ValueError: If bbox is invalid
        """
        if len(bbox) != 4:
            raise ValueError(f"Invalid bbox format: expected 4 values, got {len(bbox)}")

        width, height = image.size
        x1, y1, x2, y2 = bbox

        # Validate bbox
        if x1 >= x2 or y1 >= y2:
            raise ValueError(f"Invalid bbox dimensions: ({x1},{y1},{x2},{y2})")

        # Add padding and clamp to image bounds
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(width, x2 + padding)
        y2 = min(height, y2 + padding)

        try:
            cropped = image.crop((x1, y1, x2, y2))
            logger.debug(f"Cropped person: bbox=({x1},{y1},{x2},{y2})")
            return cropped
        except Exception as e:
            logger.error(f"Error cropping image: {e}")
            raise

    def get_detection_info(self) -> Dict:
        """
        Get current detection configuration.

        Returns:
            Dictionary with detection parameters
        """
        return {
            'device': self.device,
            'conf_threshold': self.conf_threshold,
            'min_size': self.min_size,
            'max_people': self.max_people,
            'model': str(self.model.model_name) if hasattr(self.model, 'model_name') else 'unknown'
        }
