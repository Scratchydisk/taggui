"""
Image masking utilities for removing people from scene descriptions.

This module provides functionality to mask out detected person regions in images,
allowing VLMs to focus on background/environmental elements without person contamination.
"""

import logging
from typing import List, Tuple

import numpy as np
from PIL import Image as PilImage, ImageDraw, ImageFilter

logger = logging.getLogger(__name__)


class ImageMasker:
    """
    Masks person regions in images for clean scene description generation.

    Supports multiple masking strategies:
    - 'median': Fill person regions with median background colour
    - 'blur': Apply Gaussian blur to person regions
    """

    def __init__(self, strategy: str = "median"):
        """
        Initialize the image masker.

        Args:
            strategy: Masking strategy - "median" or "blur"
        """
        if strategy not in ["median", "blur"]:
            raise ValueError(f"Invalid masking strategy: {strategy}. Must be 'median' or 'blur'")

        self.strategy = strategy
        logger.debug(f"ImageMasker initialized with strategy: {strategy}")

    def mask_persons(
        self,
        image: PilImage.Image,
        bboxes: List[Tuple[int, int, int, int]]
    ) -> PilImage.Image:
        """
        Mask person regions in the image.

        Args:
            image: PIL Image to mask
            bboxes: List of bounding boxes (x1, y1, x2, y2) for person regions

        Returns:
            Masked PIL Image with person regions obscured
        """
        if not bboxes:
            logger.debug("No bounding boxes provided, returning original image")
            return image.copy()

        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        masked_image = image.copy()

        if self.strategy == "median":
            masked_image = self._mask_with_median(masked_image, bboxes)
        elif self.strategy == "blur":
            masked_image = self._mask_with_blur(masked_image, bboxes)

        logger.debug(f"Masked {len(bboxes)} person regions using {self.strategy} strategy")
        return masked_image

    def _mask_with_median(
        self,
        image: PilImage.Image,
        bboxes: List[Tuple[int, int, int, int]]
    ) -> PilImage.Image:
        """
        Mask person regions by filling with median background colour.

        Args:
            image: PIL Image to mask
            bboxes: List of bounding boxes

        Returns:
            Masked image
        """
        img_array = np.array(image)
        height, width = img_array.shape[:2]

        # Create combined mask of all person regions
        person_mask = np.zeros((height, width), dtype=bool)
        for x1, y1, x2, y2 in bboxes:
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            person_mask[y1:y2, x1:x2] = True

        # Calculate median colour of background (non-person) regions
        if not person_mask.all():  # Check there are some background pixels
            background_pixels = img_array[~person_mask]
            if len(background_pixels) > 0:
                median_colour = np.median(background_pixels, axis=0).astype(np.uint8)
            else:
                # Fallback: use middle grey if no background pixels
                median_colour = np.array([128, 128, 128], dtype=np.uint8)
        else:
            # All pixels are person regions - use middle grey
            median_colour = np.array([128, 128, 128], dtype=np.uint8)

        # Fill person regions with median colour
        img_array[person_mask] = median_colour

        return PilImage.fromarray(img_array)

    def _mask_with_blur(
        self,
        image: PilImage.Image,
        bboxes: List[Tuple[int, int, int, int]]
    ) -> PilImage.Image:
        """
        Mask person regions by applying Gaussian blur.

        Args:
            image: PIL Image to mask
            bboxes: List of bounding boxes

        Returns:
            Masked image
        """
        # Create a heavily blurred version of the entire image
        blurred = image.filter(ImageFilter.GaussianBlur(radius=20))

        # Create a mask image (will be used to composite)
        mask_img = PilImage.new('L', image.size, 0)
        draw = ImageDraw.Draw(mask_img)

        # Draw person regions on mask
        for x1, y1, x2, y2 in bboxes:
            draw.rectangle([x1, y1, x2, y2], fill=255)

        # Smooth the mask edges for better blending
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=5))

        # Composite: use blurred version where mask is white, original where black
        masked_image = PilImage.composite(blurred, image, mask_img)

        return masked_image

    def validate_mask_quality(
        self,
        original: PilImage.Image,
        masked: PilImage.Image,
        threshold: float = 0.05
    ) -> bool:
        """
        Validate that masking was successful by checking image difference.

        Args:
            original: Original image before masking
            masked: Masked image after masking
            threshold: Minimum difference ratio to consider masking successful (0.0-1.0)

        Returns:
            True if masking created visible changes, False otherwise
        """
        # Convert to arrays
        orig_array = np.array(original.convert('RGB'))
        mask_array = np.array(masked.convert('RGB'))

        # Calculate pixel differences
        diff = np.abs(orig_array.astype(float) - mask_array.astype(float))
        total_diff = np.sum(diff)
        max_possible_diff = orig_array.size * 255  # max diff per pixel * total pixels

        diff_ratio = total_diff / max_possible_diff

        is_valid = diff_ratio >= threshold
        logger.debug(f"Mask validation: diff_ratio={diff_ratio:.4f}, threshold={threshold:.4f}, valid={is_valid}")

        return is_valid

    def visualise_mask(
        self,
        image: PilImage.Image,
        bboxes: List[Tuple[int, int, int, int]],
        show_labels: bool = True
    ) -> PilImage.Image:
        """
        Create a visualisation showing which regions will be masked.

        Args:
            image: Original PIL Image
            bboxes: List of bounding boxes that will be masked
            show_labels: Whether to show "MASKED" labels on regions

        Returns:
            Visualisation image with marked regions
        """
        # Create a copy for visualisation
        vis_image = image.copy()
        draw = ImageDraw.Draw(vis_image)

        # Draw rectangles around person regions
        for i, (x1, y1, x2, y2) in enumerate(bboxes):
            # Draw red rectangle
            draw.rectangle([x1, y1, x2, y2], outline='red', width=3)

            # Optionally add label
            if show_labels:
                label = f"MASKED {i+1}"
                # Draw text background
                text_bbox = draw.textbbox((x1, y1 - 20), label)
                draw.rectangle(text_bbox, fill='red')
                draw.text((x1, y1 - 20), label, fill='white')

        return vis_image


def mask_out_people(
    image: PilImage.Image,
    person_masks: List[np.ndarray],
    strategy: str = "median"
) -> PilImage.Image:
    """
    Convenience function to mask out people using segmentation masks.

    Args:
        image: PIL Image to mask
        person_masks: List of binary masks (numpy arrays) for each person
        strategy: Masking strategy - "median" or "blur"

    Returns:
        Masked image with people regions obscured
    """
    if not person_masks:
        return image.copy()

    # Convert segmentation masks to bounding boxes
    bboxes = []
    for mask in person_masks:
        # Find bounding box from mask
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        if not rows.any() or not cols.any():
            continue

        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]

        bboxes.append((int(x1), int(y1), int(x2), int(y2)))

    # Apply masking
    masker = ImageMasker(strategy=strategy)
    return masker.mask_persons(image, bboxes)
