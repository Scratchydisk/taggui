import sys
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image as PilImage, ImageDraw
from PySide6.QtCore import QModelIndex, Qt, QTimer, Signal, Slot
from PySide6.QtGui import QFontMetrics, QImage, QMovie, QPainter, QPen, QPixmap, QTextCursor, QWheelEvent
from PySide6.QtWidgets import (QAbstractScrollArea, QApplication, QDialog,
                               QDockWidget, QFormLayout, QFrame,
                               QGraphicsPixmapItem, QGraphicsScene,
                               QGraphicsView, QHBoxLayout, QHeaderView, QLabel,
                               QMessageBox, QPlainTextEdit, QProgressBar,
                               QPushButton, QScrollArea, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

from auto_captioning.captioning_thread import CaptioningThread
from auto_captioning.models.multi_person_tagger import MultiPersonTagger
from auto_captioning.models.wd_tagger import WdTagger
from auto_captioning.models_list import MODELS, get_model_class
from dialogs.caption_multiple_images_dialog import CaptionMultipleImagesDialog
from models.image_list_model import ImageListModel
from utils.big_widgets import TallPushButton
from utils.enums import CaptionDevice, CaptionPosition
from utils.settings import DEFAULT_SETTINGS, get_settings, get_tag_separator
from utils.settings_widgets import (FocusedScrollSettingsComboBox,
                                    FocusedScrollSettingsDoubleSpinBox,
                                    FocusedScrollSettingsSpinBox,
                                    SettingsBigCheckBox, SettingsLineEdit,
                                    SettingsPlainTextEdit)
from utils.utils import pluralize
from widgets.image_list import ImageList


def set_text_edit_height(text_edit: QPlainTextEdit, line_count: int):
    """
    Set the height of a text edit to the height of a given number of lines.
    """
    # From https://stackoverflow.com/a/46997337.
    document = text_edit.document()
    font_metrics = QFontMetrics(document.defaultFont())
    margins = text_edit.contentsMargins()
    height = int(font_metrics.lineSpacing() * line_count
                 + margins.top() + margins.bottom()
                 + document.documentMargin() * 2
                 + text_edit.frameWidth() * 2)
    text_edit.setFixedHeight(height)


class HorizontalLine(QFrame):
    def __init__(self):
        super().__init__()
        self.setFrameShape(QFrame.Shape.HLine)
        self.setFrameShadow(QFrame.Shadow.Raised)


class ZoomableGraphicsView(QGraphicsView):
    """QGraphicsView with mouse wheel zoom and pan support."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self._zoom = 0
        self._zoom_factor = 1.15
        self.edit_mode_enabled = False
        self.detection_dialog = None
        self.is_painting = False

    def wheelEvent(self, event: QWheelEvent):
        """Handle mouse wheel for zooming."""
        if event.angleDelta().y() > 0:
            # Zoom in
            factor = self._zoom_factor
            self._zoom += 1
        else:
            # Zoom out
            factor = 1 / self._zoom_factor
            self._zoom -= 1

        # Limit zoom range
        if self._zoom > 10:
            self._zoom = 10
            return
        if self._zoom < -10:
            self._zoom = -10
            return

        self.scale(factor, factor)

    def reset_zoom(self):
        """Reset zoom to fit view."""
        self.resetTransform()
        self._zoom = 0
        if self.scene() and not self.scene().sceneRect().isEmpty():
            self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def showEvent(self, event):
        """Fit view when first shown."""
        super().showEvent(event)
        if self.scene() and not self.scene().sceneRect().isEmpty():
            self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def mousePressEvent(self, event):
        """Handle mouse press for painting."""
        if self.edit_mode_enabled and self.detection_dialog and event.button() == Qt.MouseButton.LeftButton:
            # Start painting
            self.is_painting = True
            # Convert view coordinates to scene coordinates
            scene_pos = self.mapToScene(event.pos())
            self.paint_at_position(scene_pos)
        else:
            # Default behavior (panning)
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Handle mouse move for painting."""
        if self.edit_mode_enabled and self.detection_dialog and self.is_painting:
            # Continue painting
            scene_pos = self.mapToScene(event.pos())
            self.paint_at_position(scene_pos)
        else:
            # Default behavior (panning)
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """Handle mouse release."""
        if self.is_painting:
            self.is_painting = False
        super().mouseReleaseEvent(event)

    def paint_at_position(self, scene_pos):
        """Paint at the given scene position."""
        if not self.detection_dialog:
            return

        # Convert scene coordinates to image coordinates
        # Scene coordinates are in pixels relative to the displayed image
        x = int(scene_pos.x())
        y = int(scene_pos.y())

        # Get brush size from dialog
        brush_size = self.detection_dialog.brush_size_slider.value()

        # Paint on the mask
        self.detection_dialog.paint_mask_at(x, y, brush_size)


class DetectionPreviewDialog(QDialog):
    """Non-modal dialog for previewing person detection results."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Detection Preview")

        # Size dialog larger - 80% of parent or default to large size
        if parent:
            # Try to get the main window if parent is smaller
            main_window = parent
            while main_window.parent():
                main_window = main_window.parent()

            parent_width = main_window.width()
            parent_height = main_window.height()
            dialog_width = int(parent_width * 0.8)
            dialog_height = int(parent_height * 0.9)
            self.resize(dialog_width, dialog_height)
        else:
            self.resize(1400, 1000)

        self.setModal(False)  # Non-modal allows adjusting settings in parent

        # Store data
        self.image_path = None
        self.detection_settings = {}
        self.settings_form = None  # Reference to parent settings form for refresh
        self.current_detections = []
        self.current_image = None
        self.highlighted_person = None  # Index of highlighted person (None = all)

        # Main layout
        layout = QVBoxLayout(self)

        # Settings display
        self.settings_label = QLabel()
        self.settings_label.setWordWrap(True)
        layout.addWidget(self.settings_label)

        # Loading indicator (prominent, centered)
        self.loading_label = QLabel("⏳ Running detection, please wait...")
        self.loading_label.setStyleSheet(
            "color: #0066cc; font-weight: bold; font-size: 14px; "
            "background-color: #e6f2ff; padding: 10px; border-radius: 5px;"
        )
        self.loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.loading_label.hide()
        layout.addWidget(self.loading_label)

        # Zoom info layout
        zoom_info_layout = QHBoxLayout()
        self.zoom_label = QLabel("Mouse wheel to zoom, drag to pan")
        self.zoom_label.setStyleSheet("color: #666; font-style: italic;")
        zoom_info_layout.addWidget(self.zoom_label)
        zoom_info_layout.addStretch()
        layout.addLayout(zoom_info_layout)

        # Image display with zoomable graphics view
        self.graphics_view = ZoomableGraphicsView()
        self.graphics_scene = QGraphicsScene()
        self.graphics_view.setScene(self.graphics_scene)
        self.graphics_view.setMinimumHeight(500)
        layout.addWidget(self.graphics_view)

        # Detection count label
        self.detection_count_label = QLabel()
        self.detection_count_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.detection_count_label)

        # Detection info table
        self.detection_table = QTableWidget()
        self.detection_table.setColumnCount(4)
        self.detection_table.setHorizontalHeaderLabels(
            ["#", "Confidence", "Size (WxH)", "Bbox (x1,y1,x2,y2)"]
        )
        self.detection_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.detection_table.setMaximumHeight(150)
        self.detection_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.detection_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.detection_table.itemSelectionChanged.connect(self.on_person_selected)
        layout.addWidget(self.detection_table)

        # Crop previews scroll area (initially hidden)
        self.crop_previews_label = QLabel("Crop Previews (what WD Tagger will see):")
        self.crop_previews_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        self.crop_previews_label.hide()
        layout.addWidget(self.crop_previews_label)

        self.crop_previews_scroll = QScrollArea()
        self.crop_previews_scroll.setWidgetResizable(True)
        self.crop_previews_scroll.setMaximumHeight(200)
        self.crop_previews_scroll.hide()

        self.crop_previews_container = QWidget()
        self.crop_previews_layout = QHBoxLayout(self.crop_previews_container)
        self.crop_previews_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.crop_previews_scroll.setWidget(self.crop_previews_container)
        layout.addWidget(self.crop_previews_scroll)

        # Mask editing controls (initially hidden)
        self.mask_edit_container = QWidget()
        mask_edit_layout = QHBoxLayout(self.mask_edit_container)
        mask_edit_layout.setContentsMargins(0, 5, 0, 5)

        self.edit_mode_checkbox = QPushButton("Edit Masks")
        self.edit_mode_checkbox.setCheckable(True)
        self.edit_mode_checkbox.setToolTip("Enable mask editing mode to paint/erase masks")
        self.edit_mode_checkbox.clicked.connect(self.toggle_edit_mode)

        self.paint_mode_label = QLabel("Mode:")
        self.paint_radio = QPushButton("Paint")
        self.paint_radio.setCheckable(True)
        self.paint_radio.setChecked(True)
        self.erase_radio = QPushButton("Erase")
        self.erase_radio.setCheckable(True)

        # Make paint/erase mutually exclusive
        self.paint_radio.clicked.connect(lambda: self.set_brush_mode('paint'))
        self.erase_radio.clicked.connect(lambda: self.set_brush_mode('erase'))

        self.brush_size_label = QLabel("Brush size:")
        self.brush_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.brush_size_slider.setMinimum(5)
        self.brush_size_slider.setMaximum(100)
        self.brush_size_slider.setValue(20)
        self.brush_size_slider.setMaximumWidth(150)
        self.brush_size_value_label = QLabel("20px")
        self.brush_size_slider.valueChanged.connect(
            lambda v: self.brush_size_value_label.setText(f"{v}px"))

        self.selected_person_label = QLabel("No person selected")
        self.selected_person_label.setStyleSheet("font-weight: bold; color: #0066cc;")

        self.reset_masks_button = QPushButton("Reset All")
        self.reset_masks_button.setToolTip("Reset all masks to original state")
        self.reset_masks_button.clicked.connect(self.reset_masks)

        mask_edit_layout.addWidget(self.edit_mode_checkbox)
        mask_edit_layout.addWidget(self.paint_mode_label)
        mask_edit_layout.addWidget(self.paint_radio)
        mask_edit_layout.addWidget(self.erase_radio)
        mask_edit_layout.addWidget(self.brush_size_label)
        mask_edit_layout.addWidget(self.brush_size_slider)
        mask_edit_layout.addWidget(self.brush_size_value_label)
        mask_edit_layout.addWidget(self.selected_person_label)
        mask_edit_layout.addStretch()
        mask_edit_layout.addWidget(self.reset_masks_button)

        # Hide editing controls initially
        self.paint_mode_label.hide()
        self.paint_radio.hide()
        self.erase_radio.hide()
        self.brush_size_label.hide()
        self.brush_size_slider.hide()
        self.brush_size_value_label.hide()
        self.selected_person_label.hide()
        self.reset_masks_button.hide()

        self.mask_edit_container.hide()
        layout.addWidget(self.mask_edit_container)

        # Buttons
        button_layout = QHBoxLayout()
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.run_detection)
        self.fit_button = QPushButton("Fit to View")
        self.fit_button.clicked.connect(self.graphics_view.reset_zoom)
        self.show_crops_button = QPushButton("Show Crops")
        self.show_crops_button.clicked.connect(self.show_crop_previews)
        self.show_crops_button.setEnabled(False)
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.close)
        button_layout.addWidget(self.refresh_button)
        button_layout.addWidget(self.fit_button)
        button_layout.addWidget(self.show_crops_button)
        button_layout.addStretch()
        button_layout.addWidget(self.close_button)
        layout.addLayout(button_layout)

        # Initialize editing state
        self.edit_mode_enabled = False
        self.brush_mode = 'paint'  # 'paint' or 'erase'
        self.original_detections = None  # Backup of original masks
        self.is_painting = False

    def set_image_and_settings(self, image_path: str, detection_settings: dict):
        """Set the image path and detection settings, then run detection after showing dialog."""
        self.image_path = image_path
        self.detection_settings = detection_settings
        self.update_settings_display()

        # Clear previous results
        self.graphics_scene.clear()
        self.detection_count_label.setText("")
        self.detection_table.setRowCount(0)

        # Show loading indicator immediately
        self.loading_label.show()
        self.refresh_button.setEnabled(False)

        # Defer detection to allow dialog to show first
        QTimer.singleShot(100, self.run_detection)

    def update_settings_display(self):
        """Update the settings display label."""
        settings = self.detection_settings
        mask_status = "enabled" if settings.get('mask_overlapping_people', True) else "disabled"
        mask_method = settings.get('masking_method', 'Bounding box')
        iterative_status = "enabled" if settings.get('split_merged_people', True) else "disabled"
        text = (
            f"Settings: Confidence={settings.get('detection_confidence', 0.5):.2f}, "
            f"MinSize={settings.get('detection_min_size', 50)}px, "
            f"Padding={settings.get('crop_padding', 10)}px, "
            f"YOLO={settings.get('yolo_model_size', 'm')}, "
            f"Iterative={iterative_status}, "
            f"Masking={mask_status} ({mask_method})"
        )
        self.settings_label.setText(text)

    def run_detection(self):
        """Run person detection and display results."""
        if not self.image_path:
            return

        # Refresh settings from parent form if available
        if self.settings_form:
            self.detection_settings = {
                'detection_confidence': self.settings_form.detection_confidence_spin_box.value(),
                'detection_min_size': self.settings_form.detection_min_size_spin_box.value(),
                'detection_max_people': self.settings_form.detection_max_people_spin_box.value(),
                'crop_padding': self.settings_form.crop_padding_spin_box.value(),
                'yolo_model_size': self.settings_form.yolo_model_size_combo_box.currentText(),
                'split_merged_people': self.settings_form.split_merged_people_check_box.isChecked(),
                'mask_overlapping_people': self.settings_form.mask_overlaps_check_box.isChecked(),
                'masking_method': self.settings_form.masking_method_combo_box.currentText(),
                'preserve_target_bbox': self.settings_form.preserve_target_bbox_check_box.isChecked(),
                'mask_erosion_size': self.settings_form.mask_erosion_spin_box.value(),
                'mask_dilation_size': self.settings_form.mask_dilation_spin_box.value(),
                'mask_blur_size': self.settings_form.mask_blur_spin_box.value(),
            }
            self.update_settings_display()

        # Ensure loading indicator is shown and button disabled
        if not self.loading_label.isVisible():
            self.loading_label.show()
        if self.refresh_button.isEnabled():
            self.refresh_button.setEnabled(False)
        QApplication.processEvents()  # Force UI update

        try:
            # Import here to avoid circular dependencies
            from auto_captioning.utils.person_detector import PersonDetector

            # Determine if segmentation should be used
            mask_overlapping = self.detection_settings.get('mask_overlapping_people', True)
            masking_method = self.detection_settings.get('masking_method', 'Bounding box')
            split_merged_people = self.detection_settings.get('split_merged_people', True)
            use_segmentation = (mask_overlapping and masking_method == 'Segmentation') or split_merged_people

            # Create PersonDetector with current settings
            detector = PersonDetector(
                model_size=self.detection_settings.get('yolo_model_size', 'm'),
                device='cpu',  # Use CPU for preview to avoid GPU memory issues
                conf_threshold=self.detection_settings.get('detection_confidence', 0.5),
                min_size=self.detection_settings.get('detection_min_size', 50),
                max_people=self.detection_settings.get('detection_max_people', 10),
                use_segmentation=use_segmentation
            )

            # Apply split merged people if enabled
            if split_merged_people:
                from auto_captioning.models.multi_person_tagger import MultiPersonTagger
                from auto_captioning.models.wd_tagger import WdTaggerModel
                import numpy as np

                # Load image for WD Tagger
                pil_image = PilImage.open(self.image_path).convert('RGBA')

                # Initialize WD Tagger model (using default model)
                wd_model_id = self.settings_form.wd_model_combo_box.currentText()
                wd_model = WdTaggerModel(wd_model_id)

                # Preprocess for WD Tagger (simplified version)
                canvas = PilImage.new('RGBA', pil_image.size, (255, 255, 255))
                canvas.alpha_composite(pil_image)
                pil_image_rgb = canvas.convert('RGB')
                max_dimension = max(pil_image_rgb.size)
                canvas = PilImage.new('RGB', (max_dimension, max_dimension), (255, 255, 255))
                horizontal_padding = (max_dimension - pil_image_rgb.width) // 2
                vertical_padding = (max_dimension - pil_image_rgb.height) // 2
                canvas.paste(pil_image_rgb, (horizontal_padding, vertical_padding))
                _, input_dimension, *_ = wd_model.inference_session.get_inputs()[0].shape
                if max_dimension != input_dimension:
                    canvas = canvas.resize((input_dimension, input_dimension), resample=PilImage.Resampling.BICUBIC)
                image_array = np.array(canvas, dtype=np.float32)[:, :, ::-1]
                image_array = np.expand_dims(image_array, axis=0)

                # Get tags
                wd_settings = {'show_probabilities': False, 'min_probability': 0.35, 'max_tags': 50, 'tags_to_exclude': ''}
                full_image_tags, _ = wd_model.generate_tags(image_array, wd_settings)

                # Parse expected person count
                expected_person_count = MultiPersonTagger.parse_person_count_from_tags(full_image_tags)

                # Use iterative detection if we expect people
                if expected_person_count > 0:
                    detections, original_detection_count = detector.detect_people_iteratively(
                        self.image_path,
                        expected_person_count,
                        max_iterations=3
                    )
                else:
                    # No expected people, use standard detection
                    detections = detector.detect_people(self.image_path)
                    original_detection_count = len(detections)
            else:
                # Standard detection (no splitting)
                detections = detector.detect_people(self.image_path)
                original_detection_count = len(detections)

            # Load image
            pil_image = PilImage.open(self.image_path).convert('RGB')

            # Store for re-drawing when person selected
            self.current_detections = detections
            self.current_image = pil_image
            self.highlighted_person = None  # Reset highlight

            # Draw bounding boxes and masking visualization
            annotated_image = self.draw_detections(
                pil_image,
                detections,
                self.detection_settings.get('crop_padding', 10),
                mask_overlapping,
                masking_method,
                self.highlighted_person
            )

            # Convert PIL Image to QPixmap in memory (no disk I/O)
            buffer = BytesIO()
            annotated_image.save(buffer, format='PNG')
            buffer.seek(0)
            qimage = QImage.fromData(buffer.getvalue())
            pixmap = QPixmap.fromImage(qimage)

            # Clear and update scene
            self.graphics_scene.clear()
            pixmap_item = QGraphicsPixmapItem(pixmap)
            self.graphics_scene.addItem(pixmap_item)
            self.graphics_scene.setSceneRect(pixmap_item.boundingRect())

            # Fit image to view on first load or refresh (delayed to allow layout)
            QTimer.singleShot(50, self.graphics_view.reset_zoom)

            # Update detection count
            count = len(detections)
            # Check if segmentation masks are present
            masks_present = any(d.get('mask') is not None for d in detections)
            mask_info = f" (segmentation masks: {'yes' if masks_present else 'no'})" if use_segmentation else ""

            # Show iterative detection info if applied
            iteration_info = ""
            if split_merged_people and original_detection_count != count:
                iteration_info = f" [iterative: initial={original_detection_count}]"

            self.detection_count_label.setText(
                f"{count} {'person' if count == 1 else 'people'} detected{iteration_info}{mask_info}"
            )

            # Update table
            self.update_detection_table(detections)

            # Enable show crops button if we have detections
            self.show_crops_button.setEnabled(len(detections) > 0)

            # Enable mask editing if we have detections with masks
            has_masks = any(d.get('mask') is not None for d in detections)
            self.mask_edit_container.setVisible(has_masks)

        except Exception as e:
            QMessageBox.critical(
                self,
                "Detection Error",
                f"Failed to run detection: {str(e)}"
            )

        finally:
            # Hide loading indicator and re-enable button
            self.loading_label.setText("⏳ Running detection, please wait...")  # Reset text
            self.loading_label.hide()
            self.refresh_button.setEnabled(True)

    def draw_detections(
        self,
        image: PilImage.Image,
        detections: list,
        padding: int,
        mask_overlapping: bool = False,
        masking_method: str = 'Bounding box',
        highlighted_person: int = None
    ) -> PilImage.Image:
        """Draw bounding boxes, padding, and masking visualization on image.

        Args:
            highlighted_person: If set, only show masking for this person's crop region
        """
        import numpy as np

        # Make a copy to draw on
        img_copy = image.copy()

        # If masking is enabled and we have multiple people, show masking visualization
        if mask_overlapping and len(detections) > 1:
            # Apply masking visualization for each person
            img_array = np.array(img_copy)

            # If a person is highlighted, only show masking for that person
            # Otherwise show masking for all people
            target_indices = [highlighted_person] if highlighted_person is not None else range(len(detections))

            for target_idx in target_indices:
                target_detection = detections[target_idx]
                target_bbox = target_detection['bbox']
                tx1, ty1, tx2, ty2 = target_bbox

                # Calculate crop region
                crop_x1 = max(0, tx1 - padding)
                crop_y1 = max(0, ty1 - padding)
                crop_x2 = min(img_copy.width, tx2 + padding)
                crop_y2 = min(img_copy.height, ty2 + padding)

                # For each other person, show what would be masked
                for other_idx, other_detection in enumerate(detections):
                    if other_idx == target_idx:
                        continue

                    if masking_method == 'Segmentation' and 'mask' in other_detection and other_detection['mask'] is not None:
                        # Use segmentation mask
                        mask = other_detection['mask']
                        # Check if mask overlaps with crop region
                        mask_crop = mask[crop_y1:crop_y2, crop_x1:crop_x2]
                        if mask_crop.any():
                            # Semi-transparent grey overlay on masked pixels
                            img_array[mask] = img_array[mask] * 0.5 + np.array([127, 127, 127]) * 0.5
                    else:
                        # Use bounding box
                        other_bbox = other_detection['bbox']
                        ox1, oy1, ox2, oy2 = other_bbox

                        # Check if other bbox overlaps with crop region
                        if not (ox2 < crop_x1 or ox1 > crop_x2 or oy2 < crop_y1 or oy1 > crop_y2):
                            # Semi-transparent grey overlay on masked region
                            img_array[oy1:oy2, ox1:ox2] = img_array[oy1:oy2, ox1:ox2] * 0.5 + np.array([127, 127, 127]) * 0.5

            img_copy = PilImage.fromarray(img_array.astype('uint8'))

        draw = ImageDraw.Draw(img_copy)

        # Color palette for different people (RGB)
        colors = [
            (0, 255, 0),      # Green
            (0, 100, 255),    # Blue
            (255, 50, 50),    # Red
            (255, 165, 0),    # Orange
            (255, 0, 255),    # Magenta
            (0, 255, 255),    # Cyan
            (255, 255, 0),    # Yellow
            (128, 0, 128),    # Purple
            (255, 192, 203),  # Pink
            (165, 42, 42),    # Brown
        ]

        for i, detection in enumerate(detections):
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            color = colors[i % len(colors)]
            mask = detection.get('mask')

            # Determine if this is the highlighted person
            is_highlighted = (highlighted_person is not None and i == highlighted_person)

            # If segmentation mask is available, draw it
            if mask is not None and mask.shape[0] > 0 and mask.shape[1] > 0:
                # Create colored overlay showing the segmentation mask
                img_array = np.array(img_copy)

                # Create a semi-transparent colored overlay
                overlay = img_array.copy()

                # Fill mask area with the person's color (semi-transparent)
                overlay[mask] = overlay[mask] * 0.6 + np.array(color) * 0.4

                # Find mask boundary using simple numpy operations
                # Pad mask to handle edges
                padded = np.pad(mask, 1, mode='constant', constant_values=False)

                # Detect boundary by checking if any neighbor is False
                # A pixel is on the boundary if it's True but has at least one False neighbor
                boundary_pixels = (
                    padded[1:-1, 1:-1] &  # Current pixel is True
                    ~(
                        padded[:-2, 1:-1] &  # Top
                        padded[2:, 1:-1] &   # Bottom
                        padded[1:-1, :-2] &  # Left
                        padded[1:-1, 2:]     # Right
                    )
                )

                # Thicken boundary for visibility
                # Dilate boundary by including neighbors
                thick_boundary = boundary_pixels.copy()
                for _ in range(2 if not is_highlighted else 3):
                    padded_boundary = np.pad(thick_boundary, 1, mode='constant', constant_values=False)
                    thick_boundary = (
                        padded_boundary[1:-1, 1:-1] |
                        padded_boundary[:-2, 1:-1] |
                        padded_boundary[2:, 1:-1] |
                        padded_boundary[1:-1, :-2] |
                        padded_boundary[1:-1, 2:]
                    )

                # Draw boundary in the person's color
                if is_highlighted:
                    # Extra thick boundary with white glow for highlighted
                    extra_thick = thick_boundary.copy()
                    for _ in range(2):
                        padded_extra = np.pad(extra_thick, 1, mode='constant', constant_values=False)
                        extra_thick = (
                            padded_extra[1:-1, 1:-1] |
                            padded_extra[:-2, 1:-1] |
                            padded_extra[2:, 1:-1] |
                            padded_extra[1:-1, :-2] |
                            padded_extra[1:-1, 2:]
                        )
                    outer_ring = extra_thick & ~thick_boundary
                    overlay[outer_ring] = [255, 255, 255]  # White glow

                overlay[thick_boundary] = color  # Main boundary

                img_copy = PilImage.fromarray(overlay.astype('uint8'))
                draw = ImageDraw.Draw(img_copy)
            else:
                # Fall back to bounding box if no mask
                bbox_width = 8 if is_highlighted else 4
                if is_highlighted:
                    # Draw white glow around highlighted bbox
                    draw.rectangle([x1-2, y1-2, x2+2, y2+2], outline=(255, 255, 255), width=2)
                draw.rectangle([x1, y1, x2, y2], outline=color, width=bbox_width)

            # Draw padding region (thin dashed-like line)
            px1 = max(0, x1 - padding)
            py1 = max(0, y1 - padding)
            px2 = min(img_copy.width, x2 + padding)
            py2 = min(img_copy.height, y2 + padding)

            # Draw padding box with thinner line and lighter color (bolder if highlighted)
            padding_width = 4 if is_highlighted else 2
            padding_color = tuple(min(255, c + 80) for c in color)
            draw.rectangle([px1, py1, px2, py2], outline=padding_color, width=padding_width)

            # Draw person number (larger and bold if highlighted)
            text = f"Person {i+1}" + (" ★" if is_highlighted else "")
            # Simple text position - top left of bbox
            draw.text((x1 + 5, y1 + 5), text, fill=color)

        return img_copy

    def on_person_selected(self):
        """Handle person selection in detection table."""
        selected_rows = self.detection_table.selectedItems()
        if not selected_rows or not self.current_image or not self.current_detections:
            return

        # Get selected row index
        row = self.detection_table.currentRow()
        if row < 0 or row >= len(self.current_detections):
            return

        # Show loading indicator
        self.loading_label.setText(f"⏳ Updating preview for Person {row+1}...")
        self.loading_label.show()
        self.detection_table.setEnabled(False)
        QApplication.processEvents()

        # Set highlighted person and defer redraw to allow UI update
        self.highlighted_person = row

        # Update edit label if in edit mode
        if self.edit_mode_enabled:
            self.selected_person_label.setText(f"Editing: Person {row + 1}")

        QTimer.singleShot(50, self.redraw_with_highlight)

    def redraw_with_highlight(self):
        """Redraw the image with the selected person highlighted."""
        if not self.current_image or not self.current_detections:
            return

        try:
            mask_overlapping = self.detection_settings.get('mask_overlapping_people', True)
            masking_method = self.detection_settings.get('masking_method', 'Bounding box')

            # Draw with highlighted person
            annotated_image = self.draw_detections(
                self.current_image,
                self.current_detections,
                self.detection_settings.get('crop_padding', 10),
                mask_overlapping,
                masking_method,
                self.highlighted_person
            )

            # Convert and display
            buffer = BytesIO()
            annotated_image.save(buffer, format='PNG')
            buffer.seek(0)
            qimage = QImage.fromData(buffer.getvalue())
            pixmap = QPixmap.fromImage(qimage)

            # Update scene
            self.graphics_scene.clear()
            pixmap_item = QGraphicsPixmapItem(pixmap)
            self.graphics_scene.addItem(pixmap_item)
            self.graphics_scene.setSceneRect(pixmap_item.boundingRect())

        finally:
            # Hide loading indicator and re-enable table
            self.loading_label.hide()
            self.detection_table.setEnabled(True)

    def toggle_edit_mode(self):
        """Toggle mask editing mode."""
        self.edit_mode_enabled = self.edit_mode_checkbox.isChecked()

        if self.edit_mode_enabled:
            # Show editing controls
            self.paint_mode_label.show()
            self.paint_radio.show()
            self.erase_radio.show()
            self.brush_size_label.show()
            self.brush_size_slider.show()
            self.brush_size_value_label.show()
            self.selected_person_label.show()
            self.reset_masks_button.show()

            # Backup original masks
            if self.original_detections is None:
                import copy
                self.original_detections = copy.deepcopy(self.current_detections)

            # Update selected person label
            if self.highlighted_person is not None:
                self.selected_person_label.setText(f"Editing: Person {self.highlighted_person + 1}")
            else:
                self.selected_person_label.setText("Select a person from the table")

            # Enable painting on graphics view
            self.graphics_view.edit_mode_enabled = True
            self.graphics_view.detection_dialog = self
            # Disable panning to allow painting
            self.graphics_view.setDragMode(QGraphicsView.DragMode.NoDrag)
        else:
            # Hide editing controls
            self.paint_mode_label.hide()
            self.paint_radio.hide()
            self.erase_radio.hide()
            self.brush_size_label.hide()
            self.brush_size_slider.hide()
            self.brush_size_value_label.hide()
            self.selected_person_label.hide()
            self.reset_masks_button.hide()

            # Disable painting and restore panning
            self.graphics_view.edit_mode_enabled = False
            self.graphics_view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)

    def set_brush_mode(self, mode: str):
        """Set brush mode to 'paint' or 'erase'."""
        self.brush_mode = mode
        if mode == 'paint':
            self.paint_radio.setChecked(True)
            self.erase_radio.setChecked(False)
        else:
            self.paint_radio.setChecked(False)
            self.erase_radio.setChecked(True)

    def reset_masks(self):
        """Reset all masks to original state."""
        if self.original_detections is not None:
            import copy
            self.current_detections = copy.deepcopy(self.original_detections)
            self.original_detections = None

            # Re-draw with original masks
            self.redraw_with_highlight()

            QMessageBox.information(
                self,
                "Masks Reset",
                "All masks have been reset to their original state."
            )

    def paint_mask_at(self, x: int, y: int, brush_size: int):
        """Paint or erase mask at the given image coordinates."""
        if self.highlighted_person is None:
            return

        if not (0 <= self.highlighted_person < len(self.current_detections)):
            return

        detection = self.current_detections[self.highlighted_person]
        mask = detection.get('mask')

        if mask is None:
            return

        # Get brush radius
        radius = brush_size // 2

        # Create circular brush
        import cv2
        height, width = mask.shape

        # Clamp coordinates
        y = max(0, min(height - 1, y))
        x = max(0, min(width - 1, x))

        # Create circular mask for brush
        y_indices, x_indices = np.ogrid[-radius:radius+1, -radius:radius+1]
        circle_mask = x_indices**2 + y_indices**2 <= radius**2

        # Calculate brush bounds
        y_min = max(0, y - radius)
        y_max = min(height, y + radius + 1)
        x_min = max(0, x - radius)
        x_max = min(width, x + radius + 1)

        # Adjust circle mask if brush extends beyond image
        circle_y_min = max(0, radius - y)
        circle_y_max = circle_y_min + (y_max - y_min)
        circle_x_min = max(0, radius - x)
        circle_x_max = circle_x_min + (x_max - x_min)

        circle_mask_crop = circle_mask[circle_y_min:circle_y_max, circle_x_min:circle_x_max]

        # Apply paint or erase
        if self.brush_mode == 'paint':
            mask[y_min:y_max, x_min:x_max] |= circle_mask_crop
        else:  # erase
            mask[y_min:y_max, x_min:x_max] &= ~circle_mask_crop

        # Update the detection's mask
        detection['mask'] = mask

        # Re-draw to show changes
        self.redraw_with_highlight()

    def update_detection_table(self, detections: list):
        """Update the detection info table."""
        self.detection_table.setRowCount(len(detections))

        for i, detection in enumerate(detections):
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            confidence = detection['confidence']

            # Person number
            self.detection_table.setItem(i, 0, QTableWidgetItem(str(i + 1)))

            # Confidence
            self.detection_table.setItem(
                i, 1, QTableWidgetItem(f"{confidence:.3f}")
            )

            # Size
            self.detection_table.setItem(
                i, 2, QTableWidgetItem(f"{width}x{height}")
            )

            # Bbox coordinates
            self.detection_table.setItem(
                i, 3, QTableWidgetItem(f"({x1},{y1},{x2},{y2})")
            )

    def show_crop_previews(self):
        """Generate and display crop previews showing what WD Tagger will see."""
        if not self.current_detections or not self.current_image:
            return

        # Clear existing previews
        while self.crop_previews_layout.count():
            item = self.crop_previews_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Show loading indicator
        self.loading_label.setText("⏳ Generating crop previews...")
        self.loading_label.show()
        self.show_crops_button.setEnabled(False)
        QApplication.processEvents()

        try:
            from auto_captioning.models.multi_person_tagger import MultiPersonTagger

            # Get masking settings
            mask_overlapping = self.detection_settings.get('mask_overlapping_people', True)
            masking_method = self.detection_settings.get('masking_method', 'Bounding box')
            preserve_target_bbox = self.detection_settings.get('preserve_target_bbox', True)
            padding = self.detection_settings.get('crop_padding', 10)

            # Mask refinement settings
            mask_erosion = self.detection_settings.get('mask_erosion_size', 0)
            mask_dilation = self.detection_settings.get('mask_dilation_size', 0)
            mask_blur = self.detection_settings.get('mask_blur_size', 0)

            # Generate crops for each person
            for i, detection in enumerate(self.current_detections):
                # Extract person using segmentation or bbox+masking
                if masking_method == 'Segmentation' and detection.get('mask') is not None:
                    # Extract ONLY the segmented person on white background
                    cropped = self._extract_segmented_person(
                        self.current_image,
                        detection,
                        padding
                    )
                elif mask_overlapping and len(self.current_detections) > 1:
                    # Use bounding box masking (mask out other people, then crop)
                    image_to_crop = self._generate_masked_image(
                        self.current_image,
                        detection,
                        i,
                        self.current_detections,
                        padding,
                        masking_method,
                        preserve_target_bbox,
                        mask_erosion,
                        mask_dilation,
                        mask_blur
                    )
                    # Crop the person
                    bbox = detection['bbox']
                    x1, y1, x2, y2 = bbox
                    crop_x1 = max(0, x1 - padding)
                    crop_y1 = max(0, y1 - padding)
                    crop_x2 = min(image_to_crop.width, x2 + padding)
                    crop_y2 = min(image_to_crop.height, y2 + padding)
                    cropped = image_to_crop.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                else:
                    # No masking - just crop the bounding box
                    bbox = detection['bbox']
                    x1, y1, x2, y2 = bbox
                    crop_x1 = max(0, x1 - padding)
                    crop_y1 = max(0, y1 - padding)
                    crop_x2 = min(self.current_image.width, x2 + padding)
                    crop_y2 = min(self.current_image.height, y2 + padding)
                    cropped = self.current_image.crop((crop_x1, crop_y1, crop_x2, crop_y2))

                # Convert to QPixmap and display
                # Scale to max 150px height for preview
                max_height = 150
                if cropped.height > max_height:
                    aspect = cropped.width / cropped.height
                    new_height = max_height
                    new_width = int(aspect * new_height)
                    cropped = cropped.resize((new_width, new_height), PilImage.Resampling.LANCZOS)

                # Convert to QPixmap
                buffer = BytesIO()
                cropped.save(buffer, format='PNG')
                buffer.seek(0)
                qimage = QImage.fromData(buffer.getvalue())
                pixmap = QPixmap.fromImage(qimage)

                # Create label with image
                crop_widget = QWidget()
                crop_layout = QVBoxLayout(crop_widget)
                crop_layout.setContentsMargins(5, 5, 5, 5)

                image_label = QLabel()
                image_label.setPixmap(pixmap)
                image_label.setFrameStyle(QFrame.Shape.Box)

                text_label = QLabel(f"Person {i+1}")
                text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                text_label.setStyleSheet("font-weight: bold;")

                crop_layout.addWidget(image_label)
                crop_layout.addWidget(text_label)

                self.crop_previews_layout.addWidget(crop_widget)

            # Show the previews
            self.crop_previews_label.show()
            self.crop_previews_scroll.show()

        except Exception as e:
            QMessageBox.critical(
                self,
                "Crop Preview Error",
                f"Failed to generate crop previews: {str(e)}"
            )

        finally:
            # Hide loading indicator and re-enable button
            self.loading_label.hide()
            self.show_crops_button.setEnabled(True)

    def _generate_masked_image(
        self,
        image: PilImage.Image,
        target_detection: dict,
        target_index: int,
        all_detections: list,
        padding: int,
        masking_method: str,
        preserve_target_bbox: bool,
        mask_erosion: int,
        mask_dilation: int,
        mask_blur: int
    ) -> PilImage.Image:
        """Generate masked image for a person (mimics tagger's masking logic)."""
        import numpy as np
        import cv2

        # Create a copy to mask
        masked_image = image.copy()
        img_array = np.array(masked_image)

        # Calculate the crop region
        width, height = image.size
        target_bbox = target_detection['bbox']
        tx1, ty1, tx2, ty2 = target_bbox
        crop_x1 = max(0, tx1 - padding)
        crop_y1 = max(0, ty1 - padding)
        crop_x2 = min(width, tx2 + padding)
        crop_y2 = min(height, ty2 + padding)

        # Determine grey value
        num_channels = img_array.shape[2] if len(img_array.shape) == 3 else 1
        grey_value = [127] * num_channels if num_channels > 1 else 127

        # Protected region for target bbox
        protected_region = None
        if preserve_target_bbox and masking_method == 'Segmentation':
            protected_region = np.zeros((height, width), dtype=bool)
            protected_region[ty1:ty2, tx1:tx2] = True

        # Mask other people
        for i, detection in enumerate(all_detections):
            if i == target_index:
                continue

            if masking_method == 'Segmentation' and detection.get('mask') is not None:
                mask = detection['mask']

                # Apply mask refinement
                mask = self._apply_mask_refinement(
                    mask, mask_erosion, mask_dilation, mask_blur
                )

                # Create crop mask
                crop_mask = np.zeros_like(mask, dtype=bool)
                crop_mask[crop_y1:crop_y2, crop_x1:crop_x2] = True

                # Pixels to mask
                pixels_to_mask = mask & crop_mask
                if protected_region is not None:
                    pixels_to_mask = pixels_to_mask & ~protected_region

                if pixels_to_mask.any():
                    img_array[pixels_to_mask] = grey_value

            else:
                # Bounding box masking
                bbox = detection['bbox']
                x1, y1, x2, y2 = bbox
                if not (x2 < crop_x1 or x1 > crop_x2 or y2 < crop_y1 or y1 > crop_y2):
                    overlap_x1 = max(x1, crop_x1)
                    overlap_y1 = max(y1, crop_y1)
                    overlap_x2 = min(x2, crop_x2)
                    overlap_y2 = min(y2, crop_y2)
                    img_array[overlap_y1:overlap_y2, overlap_x1:overlap_x2] = grey_value

        return PilImage.fromarray(img_array)

    def _extract_segmented_person(
        self,
        image: PilImage.Image,
        detection: dict,
        padding: int
    ) -> PilImage.Image:
        """Extract only the segmented person pixels on a white background."""
        mask = detection.get('mask')
        if mask is None:
            # Fall back to regular bbox crop if no mask
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            crop_x1 = max(0, x1 - padding)
            crop_y1 = max(0, y1 - padding)
            crop_x2 = min(image.width, x2 + padding)
            crop_y2 = min(image.height, y2 + padding)
            return image.crop((crop_x1, crop_y1, crop_x2, crop_y2))

        # Get image dimensions
        width, height = image.size

        # Convert image to numpy array
        img_array = np.array(image)

        # Create white background
        num_channels = img_array.shape[2] if len(img_array.shape) == 3 else 1
        white_value = [255] * num_channels if num_channels > 1 else 255
        result_array = np.full_like(img_array, white_value)

        # Copy only the pixels where the mask is True
        result_array[mask] = img_array[mask]

        # Convert back to PIL Image
        result_image = PilImage.fromarray(result_array)

        # Find the bounding box of the mask
        mask_coords = np.argwhere(mask)
        if len(mask_coords) == 0:
            # Empty mask - fall back to bbox
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            crop_x1 = max(0, x1 - padding)
            crop_y1 = max(0, y1 - padding)
            crop_x2 = min(width, x2 + padding)
            crop_y2 = min(height, y2 + padding)
            return result_image.crop((crop_x1, crop_y1, crop_x2, crop_y2))

        # mask_coords is in (y, x) format
        y_coords = mask_coords[:, 0]
        x_coords = mask_coords[:, 1]

        # Get tight bounds
        y_min, y_max = y_coords.min(), y_coords.max()
        x_min, x_max = x_coords.min(), x_coords.max()

        # Add padding
        crop_x1 = max(0, x_min - padding)
        crop_y1 = max(0, y_min - padding)
        crop_x2 = min(width, x_max + padding + 1)
        crop_y2 = min(height, y_max + padding + 1)

        # Crop to the padded bounds
        return result_image.crop((crop_x1, crop_y1, crop_x2, crop_y2))

    def _apply_mask_refinement(
        self,
        mask: np.ndarray,
        erosion_size: int,
        dilation_size: int,
        blur_size: int
    ) -> np.ndarray:
        """Apply morphological operations to refine mask."""
        import cv2
        import numpy as np

        if erosion_size == 0 and dilation_size == 0 and blur_size == 0:
            return mask

        # Convert to uint8
        mask_uint8 = (mask.astype(np.uint8) * 255)

        # Apply erosion
        if erosion_size > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_size, erosion_size))
            mask_uint8 = cv2.erode(mask_uint8, kernel, iterations=1)

        # Apply dilation
        if dilation_size > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_size, dilation_size))
            mask_uint8 = cv2.dilate(mask_uint8, kernel, iterations=1)

        # Apply blur
        if blur_size > 0:
            kernel_size = blur_size if blur_size % 2 == 1 else blur_size + 1
            mask_uint8 = cv2.GaussianBlur(mask_uint8, (kernel_size, kernel_size), 0)

        # Convert back to boolean
        return mask_uint8 > 127


class CaptionSettingsForm(QVBoxLayout):
    def __init__(self):
        super().__init__()
        self.settings = get_settings()
        try:
            import bitsandbytes
            self.is_bitsandbytes_available = True
        except RuntimeError:
            self.is_bitsandbytes_available = False
        basic_settings_form = QFormLayout()
        basic_settings_form.setRowWrapPolicy(
            QFormLayout.RowWrapPolicy.WrapAllRows)
        basic_settings_form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        self.model_combo_box = FocusedScrollSettingsComboBox(key='model_id')
        # `setEditable()` must be called before `addItems()` to preserve any
        # custom model that was set.
        self.model_combo_box.setEditable(True)
        self.model_combo_box.addItems(self.get_local_model_paths())
        self.model_combo_box.addItems(MODELS)
        self.prompt_text_edit = SettingsPlainTextEdit(key='prompt')
        set_text_edit_height(self.prompt_text_edit, 4)
        self.caption_start_line_edit = SettingsLineEdit(key='caption_start')
        self.caption_start_line_edit.setClearButtonEnabled(True)
        self.caption_position_combo_box = FocusedScrollSettingsComboBox(
            key='caption_position')
        self.caption_position_combo_box.addItems(list(CaptionPosition))
        self.device_combo_box = FocusedScrollSettingsComboBox(key='device')
        self.device_combo_box.addItems(list(CaptionDevice))
        self.load_in_4_bit_container = QWidget()
        load_in_4_bit_layout = QHBoxLayout()
        load_in_4_bit_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        load_in_4_bit_layout.setContentsMargins(0, 0, 0, 0)
        self.load_in_4_bit_check_box = SettingsBigCheckBox(
            key='load_in_4_bit', default=True)
        load_in_4_bit_layout.addWidget(QLabel('Load in 4-bit'))
        load_in_4_bit_layout.addWidget(self.load_in_4_bit_check_box)
        self.load_in_4_bit_container.setLayout(load_in_4_bit_layout)
        self.remove_tag_separators_container = QWidget()
        remove_tag_separators_layout = QHBoxLayout(
            self.remove_tag_separators_container)
        remove_tag_separators_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        remove_tag_separators_layout.setContentsMargins(0, 0, 0, 0)
        self.remove_tag_separators_check_box = SettingsBigCheckBox(
            key='remove_tag_separators', default=True)
        remove_tag_separators_label = QLabel(
            'Remove tag separators in caption')
        remove_tag_separators_layout.addWidget(remove_tag_separators_label)
        remove_tag_separators_layout.addWidget(
            self.remove_tag_separators_check_box)
        basic_settings_form.addRow('Model', self.model_combo_box)
        self.prompt_label = QLabel('Prompt')
        basic_settings_form.addRow(self.prompt_label, self.prompt_text_edit)
        self.caption_start_label = QLabel('Start caption with')
        basic_settings_form.addRow(self.caption_start_label,
                                   self.caption_start_line_edit)
        basic_settings_form.addRow('Caption position',
                                   self.caption_position_combo_box)
        self.device_label = QLabel('Device')
        basic_settings_form.addRow(self.device_label, self.device_combo_box)
        basic_settings_form.addRow(self.load_in_4_bit_container)
        basic_settings_form.addRow(self.remove_tag_separators_container)

        self.wd_tagger_settings_form_container = QWidget()
        wd_tagger_settings_form = QFormLayout(
            self.wd_tagger_settings_form_container)
        wd_tagger_settings_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        wd_tagger_settings_form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        self.show_probabilities_check_box = SettingsBigCheckBox(
            key='wd_tagger_show_probabilities', default=True)
        self.min_probability_spin_box = FocusedScrollSettingsDoubleSpinBox(
            key='wd_tagger_min_probability', default=0.4, minimum=0.01,
            maximum=1)
        self.min_probability_spin_box.setSingleStep(0.01)
        self.max_tags_spin_box = FocusedScrollSettingsSpinBox(
            key='wd_tagger_max_tags', default=30, minimum=1, maximum=999)
        tags_to_exclude_form = QFormLayout()
        tags_to_exclude_form.setRowWrapPolicy(
            QFormLayout.RowWrapPolicy.WrapAllRows)
        tags_to_exclude_form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        self.tags_to_exclude_text_edit = SettingsPlainTextEdit(
            key='wd_tagger_tags_to_exclude')
        tags_to_exclude_form.addRow('Tags to exclude',
                                    self.tags_to_exclude_text_edit)
        set_text_edit_height(self.tags_to_exclude_text_edit, 4)
        wd_tagger_settings_form.addRow('Show probabilities',
                                       self.show_probabilities_check_box)
        wd_tagger_settings_form.addRow('Minimum probability',
                                       self.min_probability_spin_box)
        wd_tagger_settings_form.addRow('Maximum tags', self.max_tags_spin_box)
        wd_tagger_settings_form.addRow(tags_to_exclude_form)

        # Multi-person tagger settings
        self.multi_person_settings_form_container = QWidget()
        multi_person_settings_form = QFormLayout(
            self.multi_person_settings_form_container)
        multi_person_settings_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        multi_person_settings_form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self.detection_confidence_spin_box = FocusedScrollSettingsDoubleSpinBox(
            key='detection_confidence', default=0.5, minimum=0.1, maximum=1.0)
        self.detection_confidence_spin_box.setSingleStep(0.05)

        self.detection_min_size_spin_box = FocusedScrollSettingsSpinBox(
            key='detection_min_size', default=50, minimum=10, maximum=500)

        self.detection_max_people_spin_box = FocusedScrollSettingsSpinBox(
            key='detection_max_people', default=10, minimum=1, maximum=50)

        self.crop_padding_spin_box = FocusedScrollSettingsSpinBox(
            key='crop_padding', default=10, minimum=0, maximum=100)

        self.yolo_model_size_combo_box = FocusedScrollSettingsComboBox(
            key='yolo_model_size')
        self.yolo_model_size_combo_box.addItems(['n', 's', 'm', 'l', 'x'])
        self.yolo_model_size_combo_box.setCurrentText('m')

        self.split_merged_people_check_box = SettingsBigCheckBox(
            key='split_merged_people', default=True)
        self.split_merged_people_check_box.setToolTip(
            'Attempt to detect occluded people when YOLO detects fewer people than expected.\n'
            'Uses full-image tagging to determine expected person count, then iteratively\n'
            'masks out detected people and re-runs YOLO to find remaining people.')

        self.mask_overlaps_check_box = SettingsBigCheckBox(
            key='mask_overlapping_people', default=True)
        self.mask_overlaps_check_box.setToolTip(
            'Mask out other detected people when cropping each person to prevent confused tags')

        self.masking_method_combo_box = FocusedScrollSettingsComboBox(
            key='masking_method')
        self.masking_method_combo_box.addItems(['Bounding box', 'Segmentation'])
        self.masking_method_combo_box.setCurrentText('Segmentation')
        self.masking_method_combo_box.setToolTip(
            'Bounding box: Masks out other people using rectangles, then crops.\n'
            'Segmentation: Extracts ONLY the target person\'s pixels on white background.\n'
            'Segmentation is best for overlapping people - prevents tag contamination.')

        self.preserve_target_bbox_check_box = SettingsBigCheckBox(
            key='preserve_target_bbox', default=True)
        self.preserve_target_bbox_check_box.setToolTip(
            'Only applies to Bounding box masking method.\n'
            'Preserves the full bounding box area for the target person when masking others.\n'
            'Not used with Segmentation method (which extracts only target pixels).')

        self.include_scene_tags_check_box = SettingsBigCheckBox(
            key='include_scene_tags', default=True)

        self.max_scene_tags_spin_box = FocusedScrollSettingsSpinBox(
            key='max_scene_tags', default=20, minimum=1, maximum=100)

        self.max_tags_per_person_spin_box = FocusedScrollSettingsSpinBox(
            key='max_tags_per_person', default=20, minimum=5, maximum=200)

        # Use nested form layout for full-width person aliases field
        person_aliases_form = QFormLayout()
        person_aliases_form.setRowWrapPolicy(
            QFormLayout.RowWrapPolicy.WrapAllRows)
        person_aliases_form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        self.person_aliases_line_edit = SettingsLineEdit(
            key='person_aliases')
        self.person_aliases_line_edit.setPlaceholderText(
            'e.g., singer, guitarist, drummer (leave empty for person1, person2, etc.)')
        self.person_aliases_line_edit.setClearButtonEnabled(True)
        person_aliases_form.addRow('Person aliases',
                                   self.person_aliases_line_edit)

        # Preview detection button
        self.preview_detection_button = TallPushButton('Preview Detection')
        self.preview_detection_button.setToolTip(
            'Preview person detection and bounding boxes with current settings (only enabled when 1 image selected)')
        self.preview_detection_button.clicked.connect(self.show_detection_preview)
        self.preview_detection_button.setEnabled(False)  # Disabled by default

        # Use nested form layout for WD Tagger model dropdown (label on top)
        wd_model_form = QFormLayout()
        wd_model_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        wd_model_form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        self.wd_model_combo_box = FocusedScrollSettingsComboBox(
            key='wd_model')
        self.wd_model_combo_box.setEditable(True)
        # Add WD Tagger models to dropdown
        wd_models = [m for m in MODELS if 'wd' in m.lower() and 'tagger' in m.lower()]
        self.wd_model_combo_box.addItems(wd_models)
        self.wd_model_combo_box.setCurrentText('SmilingWolf/wd-eva02-large-tagger-v3')
        wd_model_form.addRow('WD Tagger model', self.wd_model_combo_box)

        # Basic multi-person settings (most commonly adjusted)
        multi_person_settings_form.addRow('Detection confidence',
                                          self.detection_confidence_spin_box)
        multi_person_settings_form.addRow('Maximum people',
                                          self.detection_max_people_spin_box)
        multi_person_settings_form.addRow(person_aliases_form)
        multi_person_settings_form.addRow('Include scene tags',
                                          self.include_scene_tags_check_box)
        multi_person_settings_form.addRow(self.preview_detection_button)
        multi_person_settings_form.addRow(wd_model_form)

        # Advanced multi-person settings toggle button
        self.toggle_advanced_mp_settings_button = TallPushButton(
            'Show Advanced Settings')

        # Advanced multi-person settings container
        self.advanced_mp_settings_container = QWidget()
        advanced_mp_settings_form = QFormLayout(
            self.advanced_mp_settings_container)
        advanced_mp_settings_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        advanced_mp_settings_form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        # ==================== PHASE 1: PERSON DETECTION ====================
        phase1_label = QLabel('Phase 1: Person Detection')
        phase1_label.setStyleSheet('font-weight: bold; color: #0066cc;')
        phase1_desc = QLabel('Finds people using YOLO')
        phase1_desc.setStyleSheet('color: #666; font-style: italic; font-size: 10px;')
        advanced_mp_settings_form.addRow(phase1_label)
        advanced_mp_settings_form.addRow(phase1_desc)

        advanced_mp_settings_form.addRow('Minimum person size (px)',
                                          self.detection_min_size_spin_box)
        advanced_mp_settings_form.addRow('YOLO model size',
                                          self.yolo_model_size_combo_box)
        advanced_mp_settings_form.addRow('Detect occluded people',
                                          self.split_merged_people_check_box)

        # ==================== PHASE 2: CROPPING & MASKING ====================
        advanced_mp_settings_form.addRow(HorizontalLine())
        phase2_label = QLabel('Phase 2: Cropping & Masking')
        phase2_label.setStyleSheet('font-weight: bold; color: #0066cc;')
        phase2_desc = QLabel('Prepares each person for tagging')
        phase2_desc.setStyleSheet('color: #666; font-style: italic; font-size: 10px;')
        advanced_mp_settings_form.addRow(phase2_label)
        advanced_mp_settings_form.addRow(phase2_desc)

        advanced_mp_settings_form.addRow('Crop padding (px)',
                                          self.crop_padding_spin_box)
        advanced_mp_settings_form.addRow('Mask overlapping people',
                                          self.mask_overlaps_check_box)

        # Indented sub-settings for masking
        advanced_mp_settings_form.addRow('  └─ Masking method',
                                          self.masking_method_combo_box)
        advanced_mp_settings_form.addRow('  └─ Preserve target bbox',
                                          self.preserve_target_bbox_check_box)

        # Experimental mask refinement (nested under masking)
        experimental_label = QLabel('  └─ Experimental Mask Refinement')
        experimental_label.setStyleSheet('font-weight: bold; color: #ff8800; font-size: 10px;')
        advanced_mp_settings_form.addRow(experimental_label)

        self.mask_erosion_spin_box = FocusedScrollSettingsSpinBox(
            key='mask_erosion_size', default=0, minimum=0, maximum=50)
        self.mask_erosion_spin_box.setToolTip(
            'Shrink masks by N pixels before masking other people.\n'
            'Prevents mask bleeding in overlapping regions. Try 3-10 pixels.')

        self.mask_dilation_spin_box = FocusedScrollSettingsSpinBox(
            key='mask_dilation_size', default=0, minimum=0, maximum=50)
        self.mask_dilation_spin_box.setToolTip(
            'Expand masks by N pixels before masking other people.\n'
            'Increases masking coverage. Use if tags from other people leak through.')

        self.mask_blur_spin_box = FocusedScrollSettingsSpinBox(
            key='mask_blur_size', default=0, minimum=0, maximum=50)
        self.mask_blur_spin_box.setToolTip(
            'Apply Gaussian blur to mask edges (kernel size).\n'
            'Softens mask boundaries. Must be odd number (will be adjusted).')

        advanced_mp_settings_form.addRow('     • Erosion (px)',
                                          self.mask_erosion_spin_box)
        advanced_mp_settings_form.addRow('     • Dilation (px)',
                                          self.mask_dilation_spin_box)
        advanced_mp_settings_form.addRow('     • Blur (px)',
                                          self.mask_blur_spin_box)

        # ==================== PHASE 3: TAGGING PARAMETERS ====================
        advanced_mp_settings_form.addRow(HorizontalLine())
        phase3_label = QLabel('Phase 3: Tagging Parameters')
        phase3_label.setStyleSheet('font-weight: bold; color: #0066cc;')
        phase3_desc = QLabel('WD Tagger settings for people & scene')
        phase3_desc.setStyleSheet('color: #666; font-style: italic; font-size: 10px;')
        advanced_mp_settings_form.addRow(phase3_label)
        advanced_mp_settings_form.addRow(phase3_desc)

        advanced_mp_settings_form.addRow('Maximum tags per person',
                                          self.max_tags_per_person_spin_box)

        # WD Tagger advanced settings for multi-person
        self.mp_min_probability_spin_box = FocusedScrollSettingsDoubleSpinBox(
            key='mp_wd_tagger_min_probability', default=0.35, minimum=0.01,
            maximum=1)
        self.mp_min_probability_spin_box.setSingleStep(0.01)

        advanced_mp_settings_form.addRow('Tag confidence threshold',
                                          self.mp_min_probability_spin_box)

        # Use nested form layout for full-width tags to exclude field
        mp_tags_to_exclude_form = QFormLayout()
        mp_tags_to_exclude_form.setRowWrapPolicy(
            QFormLayout.RowWrapPolicy.WrapAllRows)
        mp_tags_to_exclude_form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        self.mp_tags_to_exclude_text_edit = SettingsPlainTextEdit(
            key='mp_wd_tagger_tags_to_exclude')
        set_text_edit_height(self.mp_tags_to_exclude_text_edit, 4)
        mp_tags_to_exclude_form.addRow('Tags to exclude',
                                       self.mp_tags_to_exclude_text_edit)

        advanced_mp_settings_form.addRow(mp_tags_to_exclude_form)

        advanced_mp_settings_form.addRow('Maximum scene tags',
                                          self.max_scene_tags_spin_box)

        # Hide advanced settings by default
        self.advanced_mp_settings_container.hide()

        # Add toggle button and container to main form
        multi_person_settings_form.addRow(self.toggle_advanced_mp_settings_button)
        multi_person_settings_form.addRow(self.advanced_mp_settings_container)

        self.toggle_advanced_settings_form_button = TallPushButton(
            'Show Advanced Settings')

        self.advanced_settings_form_container = QWidget()
        advanced_settings_form = QFormLayout(
            self.advanced_settings_form_container)
        advanced_settings_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        advanced_settings_form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        bad_forced_words_form = QFormLayout()
        bad_forced_words_form.setRowWrapPolicy(
            QFormLayout.RowWrapPolicy.WrapAllRows)
        bad_forced_words_form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        self.bad_words_line_edit = SettingsLineEdit(key='bad_words')
        self.bad_words_line_edit.setClearButtonEnabled(True)
        self.forced_words_line_edit = SettingsLineEdit(key='forced_words')
        self.forced_words_line_edit.setClearButtonEnabled(True)
        bad_forced_words_form.addRow('Discourage from caption',
                                     self.bad_words_line_edit)
        bad_forced_words_form.addRow('Include in caption',
                                     self.forced_words_line_edit)
        self.min_new_token_count_spin_box = FocusedScrollSettingsSpinBox(
            key='min_new_tokens', default=1, minimum=1, maximum=999)
        self.max_new_token_count_spin_box = FocusedScrollSettingsSpinBox(
            key='max_new_tokens', default=100, minimum=1, maximum=999)
        self.beam_count_spin_box = FocusedScrollSettingsSpinBox(
            key='num_beams', default=1, minimum=1, maximum=99)
        self.length_penalty_spin_box = FocusedScrollSettingsDoubleSpinBox(
            key='length_penalty', default=1, minimum=-5, maximum=5)
        self.length_penalty_spin_box.setSingleStep(0.1)
        self.use_sampling_check_box = SettingsBigCheckBox(key='do_sample',
                                                          default=False)
        # The temperature must be positive.
        self.temperature_spin_box = FocusedScrollSettingsDoubleSpinBox(
            key='temperature', default=1, minimum=0.01, maximum=2)
        self.temperature_spin_box.setSingleStep(0.01)
        self.top_k_spin_box = FocusedScrollSettingsSpinBox(
            key='top_k', default=50, minimum=0, maximum=200)
        self.top_p_spin_box = FocusedScrollSettingsDoubleSpinBox(
            key='top_p', default=1, minimum=0, maximum=1)
        self.top_p_spin_box.setSingleStep(0.01)
        self.repetition_penalty_spin_box = FocusedScrollSettingsDoubleSpinBox(
            key='repetition_penalty', default=1, minimum=1, maximum=2)
        self.repetition_penalty_spin_box.setSingleStep(0.01)
        self.no_repeat_ngram_size_spin_box = FocusedScrollSettingsSpinBox(
            key='no_repeat_ngram_size', default=3, minimum=0, maximum=5)
        self.gpu_index_spin_box = FocusedScrollSettingsSpinBox(
            key='gpu_index', default=0, minimum=0, maximum=9)
        advanced_settings_form.addRow(bad_forced_words_form)
        advanced_settings_form.addRow(HorizontalLine())
        advanced_settings_form.addRow('Minimum tokens',
                                      self.min_new_token_count_spin_box)
        advanced_settings_form.addRow('Maximum tokens',
                                      self.max_new_token_count_spin_box)
        advanced_settings_form.addRow('Number of beams',
                                      self.beam_count_spin_box)
        advanced_settings_form.addRow('Length penalty',
                                      self.length_penalty_spin_box)
        advanced_settings_form.addRow('Use sampling',
                                      self.use_sampling_check_box)
        advanced_settings_form.addRow('Temperature',
                                      self.temperature_spin_box)
        advanced_settings_form.addRow('Top-k', self.top_k_spin_box)
        advanced_settings_form.addRow('Top-p', self.top_p_spin_box)
        advanced_settings_form.addRow('Repetition penalty',
                                      self.repetition_penalty_spin_box)
        advanced_settings_form.addRow('No repeat n-gram size',
                                      self.no_repeat_ngram_size_spin_box)
        advanced_settings_form.addRow(HorizontalLine())
        advanced_settings_form.addRow('GPU index', self.gpu_index_spin_box)
        self.advanced_settings_form_container.hide()

        self.addLayout(basic_settings_form)
        self.addWidget(self.wd_tagger_settings_form_container)
        self.addWidget(self.multi_person_settings_form_container)
        self.horizontal_line = HorizontalLine()
        self.addWidget(self.horizontal_line)
        self.addWidget(self.toggle_advanced_settings_form_button)
        self.addWidget(self.advanced_settings_form_container)
        self.addStretch()

        self.model_combo_box.currentTextChanged.connect(
            self.show_settings_for_model)
        self.device_combo_box.currentTextChanged.connect(
            self.set_load_in_4_bit_visibility)
        self.toggle_advanced_settings_form_button.clicked.connect(
            self.toggle_advanced_settings_form)
        self.toggle_advanced_mp_settings_button.clicked.connect(
            self.toggle_advanced_mp_settings)
        # Make sure the minimum new token count is less than or equal to the
        # maximum new token count.
        self.min_new_token_count_spin_box.valueChanged.connect(
            self.max_new_token_count_spin_box.setMinimum)
        self.max_new_token_count_spin_box.valueChanged.connect(
            self.min_new_token_count_spin_box.setMaximum)

        self.show_settings_for_model(self.model_combo_box.currentText())
        self.set_load_in_4_bit_visibility(self.device_combo_box.currentText())
        if not self.is_bitsandbytes_available:
            self.load_in_4_bit_check_box.setChecked(False)

    def get_local_model_paths(self) -> list[str]:
        models_directory_path = self.settings.value(
            'models_directory_path',
            defaultValue=DEFAULT_SETTINGS['models_directory_path'], type=str)
        if not models_directory_path:
            return []
        models_directory_path = Path(models_directory_path)
        print(f'Loading local auto-captioning model paths under '
              f'{models_directory_path}...')
        # Auto-captioning models have a `config.json` file.
        config_paths = set(models_directory_path.glob('**/config.json'))
        # WD Tagger models have a `selected_tags.csv` file.
        selected_tags_paths = set(
            models_directory_path.glob('**/selected_tags.csv'))
        model_directory_paths = [str(path.parent) for path
                                 in config_paths | selected_tags_paths]
        model_directory_paths.sort()
        print(f'Loaded {len(model_directory_paths)} model '
              f'{pluralize("path", len(model_directory_paths))}.')
        return model_directory_paths

    @Slot(str)
    def show_settings_for_model(self, model_id: str):
        model_class = get_model_class(model_id)
        is_wd_tagger_model = model_class == WdTagger
        is_multi_person_model = model_class == MultiPersonTagger

        # WD Tagger specific widgets
        wd_tagger_widgets = [self.wd_tagger_settings_form_container]

        # Multi-person tagger specific widgets
        multi_person_widgets = [self.multi_person_settings_form_container]

        # Common widgets for standard VLM models (not WD Tagger or MultiPerson)
        vlm_widgets = [
            self.prompt_label,
            self.prompt_text_edit,
            self.caption_start_label,
            self.caption_start_line_edit,
            self.device_label,
            self.device_combo_box,
            self.load_in_4_bit_container,
            self.remove_tag_separators_container,
            self.horizontal_line,
            self.toggle_advanced_settings_form_button,
            self.advanced_settings_form_container
        ]

        # Show/hide based on model type
        for widget in wd_tagger_widgets:
            widget.setVisible(is_wd_tagger_model)
        for widget in multi_person_widgets:
            widget.setVisible(is_multi_person_model)
        for widget in vlm_widgets:
            widget.setVisible(not is_wd_tagger_model and not is_multi_person_model)

        self.set_load_in_4_bit_visibility(self.device_combo_box.currentText())

    @Slot(str)
    def set_load_in_4_bit_visibility(self, device: str):
        model_id = self.model_combo_box.currentText()
        model_class = get_model_class(model_id)
        # WD Tagger and MultiPersonTagger don't support 4-bit quantization
        if model_class in (WdTagger, MultiPersonTagger):
            self.load_in_4_bit_container.setVisible(False)
            return
        is_load_in_4_bit_available = (self.is_bitsandbytes_available
                                      and device == CaptionDevice.GPU)
        self.load_in_4_bit_container.setVisible(is_load_in_4_bit_available)

    @Slot()
    def toggle_advanced_settings_form(self):
        if self.advanced_settings_form_container.isHidden():
            self.advanced_settings_form_container.show()
            self.toggle_advanced_settings_form_button.setText(
                'Hide Advanced Settings')
        else:
            self.advanced_settings_form_container.hide()
            self.toggle_advanced_settings_form_button.setText(
                'Show Advanced Settings')

    @Slot()
    def toggle_advanced_mp_settings(self):
        if self.advanced_mp_settings_container.isHidden():
            self.advanced_mp_settings_container.show()
            self.toggle_advanced_mp_settings_button.setText(
                'Hide Advanced Settings')
        else:
            self.advanced_mp_settings_container.hide()
            self.toggle_advanced_mp_settings_button.setText(
                'Show Advanced Settings')

    def get_caption_settings(self) -> dict:
        return {
            'model_id': self.model_combo_box.currentText(),
            'prompt': self.prompt_text_edit.toPlainText(),
            'caption_start': self.caption_start_line_edit.text(),
            'caption_position': self.caption_position_combo_box.currentText(),
            'device': self.device_combo_box.currentText(),
            'gpu_index': self.gpu_index_spin_box.value(),
            'load_in_4_bit': self.load_in_4_bit_check_box.isChecked(),
            'remove_tag_separators':
                self.remove_tag_separators_check_box.isChecked(),
            'bad_words': self.bad_words_line_edit.text(),
            'forced_words': self.forced_words_line_edit.text(),
            'generation_parameters': {
                'min_new_tokens': self.min_new_token_count_spin_box.value(),
                'max_new_tokens': self.max_new_token_count_spin_box.value(),
                'num_beams': self.beam_count_spin_box.value(),
                'length_penalty': self.length_penalty_spin_box.value(),
                'do_sample': self.use_sampling_check_box.isChecked(),
                'temperature': self.temperature_spin_box.value(),
                'top_k': self.top_k_spin_box.value(),
                'top_p': self.top_p_spin_box.value(),
                'repetition_penalty': self.repetition_penalty_spin_box.value(),
                'no_repeat_ngram_size':
                    self.no_repeat_ngram_size_spin_box.value()
            },
            'wd_tagger_settings': {
                'show_probabilities':
                    self.show_probabilities_check_box.isChecked(),
                'min_probability': self.min_probability_spin_box.value(),
                'max_tags': self.max_tags_spin_box.value(),
                'tags_to_exclude':
                    self.tags_to_exclude_text_edit.toPlainText()
            },
            # Multi-person tagger settings
            'detection_confidence': self.detection_confidence_spin_box.value(),
            'detection_min_size': self.detection_min_size_spin_box.value(),
            'detection_max_people': self.detection_max_people_spin_box.value(),
            'crop_padding': self.crop_padding_spin_box.value(),
            'yolo_model_size': self.yolo_model_size_combo_box.currentText(),
            'split_merged_people': self.split_merged_people_check_box.isChecked(),
            'mask_overlapping_people': self.mask_overlaps_check_box.isChecked(),
            'masking_method': self.masking_method_combo_box.currentText(),
            'preserve_target_bbox': self.preserve_target_bbox_check_box.isChecked(),
            'include_scene_tags': self.include_scene_tags_check_box.isChecked(),
            'max_scene_tags': self.max_scene_tags_spin_box.value(),
            'max_tags_per_person': self.max_tags_per_person_spin_box.value(),
            'person_aliases': self.person_aliases_line_edit.text(),
            'wd_model': self.wd_model_combo_box.currentText(),
            'mp_wd_tagger_min_probability': self.mp_min_probability_spin_box.value(),
            'mp_wd_tagger_tags_to_exclude':
                self.mp_tags_to_exclude_text_edit.toPlainText(),
            # Experimental mask refinement
            'mask_erosion_size': self.mask_erosion_spin_box.value(),
            'mask_dilation_size': self.mask_dilation_spin_box.value(),
            'mask_blur_size': self.mask_blur_spin_box.value()
        }

    def show_detection_preview(self):
        """Show the detection preview dialog."""
        # Get currently selected image
        parent = self.parent()
        while parent and not isinstance(parent, AutoCaptioner):
            parent = parent.parent()

        if not parent:
            return

        selected_indexes = parent.image_list.list_view.selectionModel().selectedIndexes()
        if len(selected_indexes) != 1:
            QMessageBox.warning(
                self.parentWidget(),
                "No Image Selected",
                "Please select exactly one image to preview detection."
            )
            return

        # Get image path
        image_index = selected_indexes[0]
        image = parent.image_list_model.data(image_index, Qt.ItemDataRole.UserRole)
        image_path = str(image.path)

        # Get current detection settings
        detection_settings = {
            'detection_confidence': self.detection_confidence_spin_box.value(),
            'detection_min_size': self.detection_min_size_spin_box.value(),
            'detection_max_people': self.detection_max_people_spin_box.value(),
            'crop_padding': self.crop_padding_spin_box.value(),
            'yolo_model_size': self.yolo_model_size_combo_box.currentText(),
            'mask_overlapping_people': self.mask_overlaps_check_box.isChecked(),
            'masking_method': self.masking_method_combo_box.currentText(),
            'preserve_target_bbox': self.preserve_target_bbox_check_box.isChecked(),
            'mask_erosion_size': self.mask_erosion_spin_box.value(),
            'mask_dilation_size': self.mask_dilation_spin_box.value(),
            'mask_blur_size': self.mask_blur_spin_box.value(),
        }

        # Create or show dialog
        if not hasattr(self, '_detection_preview_dialog') or not self._detection_preview_dialog:
            self._detection_preview_dialog = DetectionPreviewDialog(self.parentWidget())

        # Set reference to settings form for refresh
        self._detection_preview_dialog.settings_form = self

        self._detection_preview_dialog.set_image_and_settings(image_path, detection_settings)
        self._detection_preview_dialog.show()
        self._detection_preview_dialog.raise_()
        self._detection_preview_dialog.activateWindow()


@Slot()
def restore_stdout_and_stderr():
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__


class AutoCaptioner(QDockWidget):
    caption_generated = Signal(QModelIndex, str, list)

    def __init__(self, image_list_model: ImageListModel,
                 image_list: ImageList):
        super().__init__()
        self.image_list_model = image_list_model
        self.image_list = image_list
        self.settings = get_settings()
        self.is_captioning = False
        self.captioning_thread = None
        self.processor = None
        self.model = None
        self.model_id: str | None = None
        self.model_device_type: str | None = None
        self.is_model_loaded_in_4_bit = None
        # Whether the last block of text in the console text edit should be
        # replaced with the next block of text that is outputted.
        self.replace_last_console_text_edit_block = False

        # Each `QDockWidget` needs a unique object name for saving its state.
        self.setObjectName('auto_captioner')
        self.setWindowTitle('Auto-Captioner')
        self.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea
                             | Qt.DockWidgetArea.RightDockWidgetArea)

        self.start_cancel_button = TallPushButton('Start Auto-Captioning')
        self.progress_bar = QProgressBar()
        self.progress_bar.setFormat('%v / %m images captioned (%p%)')
        self.progress_bar.hide()
        self.console_text_edit = QPlainTextEdit()
        set_text_edit_height(self.console_text_edit, 4)
        self.console_text_edit.setReadOnly(True)
        self.console_text_edit.hide()
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(self.start_cancel_button)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.console_text_edit)
        self.caption_settings_form = CaptionSettingsForm()
        layout.addLayout(self.caption_settings_form)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setSizeAdjustPolicy(
            QAbstractScrollArea.SizeAdjustPolicy.AdjustToContents)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setWidget(container)
        self.setWidget(scroll_area)

        self.start_cancel_button.clicked.connect(
            self.start_or_cancel_captioning)

        # Connect to selection changes to update preview button state
        self.image_list.list_view.selectionModel().selectionChanged.connect(
            self.update_preview_button_state)

        # Also connect to model changes
        self.caption_settings_form.model_combo_box.currentTextChanged.connect(
            self.update_preview_button_state)

        # Initial update
        self.update_preview_button_state()

    @Slot()
    def update_preview_button_state(self):
        """Enable preview button only when 1 image selected and multi-person model selected."""
        selected_count = len(self.image_list.list_view.selectionModel().selectedIndexes())
        model_id = self.caption_settings_form.model_combo_box.currentText()
        is_multi_person_model = 'multi-person' in model_id.lower()

        should_enable = (selected_count == 1) and is_multi_person_model
        self.caption_settings_form.preview_detection_button.setEnabled(should_enable)

    @Slot()
    def start_or_cancel_captioning(self):
        if self.is_captioning:
            # Cancel captioning.
            self.captioning_thread.is_canceled = True
            self.start_cancel_button.setEnabled(False)
            self.start_cancel_button.setText('Canceling Auto-Captioning...')
        else:
            # Start captioning.
            self.generate_captions()

    def set_is_captioning(self, is_captioning: bool):
        self.is_captioning = is_captioning
        button_text = ('Cancel Auto-Captioning' if is_captioning
                       else 'Start Auto-Captioning')
        self.start_cancel_button.setText(button_text)

    @Slot(str)
    def update_console_text_edit(self, text: str):
        # '\x1b[A' is the ANSI escape sequence for moving the cursor up.
        if text == '\x1b[A':
            self.replace_last_console_text_edit_block = True
            return
        text = text.strip()
        if not text:
            return
        if self.console_text_edit.isHidden():
            self.console_text_edit.show()
        if self.replace_last_console_text_edit_block:
            self.replace_last_console_text_edit_block = False
            # Select and remove the last block of text.
            self.console_text_edit.moveCursor(QTextCursor.MoveOperation.End)
            self.console_text_edit.moveCursor(
                QTextCursor.MoveOperation.StartOfBlock,
                QTextCursor.MoveMode.KeepAnchor)
            self.console_text_edit.textCursor().removeSelectedText()
            # Delete the newline.
            self.console_text_edit.textCursor().deletePreviousChar()
        self.console_text_edit.appendPlainText(text)

    @Slot()
    def show_alert(self):
        if self.captioning_thread.is_canceled:
            return
        if self.captioning_thread.is_error:
            icon = QMessageBox.Icon.Critical
            text = ('An error occurred during captioning. See the '
                    'Auto-Captioner console for more information.')
        else:
            icon = QMessageBox.Icon.Information
            text = 'Captioning has finished.'
        alert = QMessageBox()
        alert.setIcon(icon)
        alert.setText(text)
        alert.exec()

    @Slot()
    def generate_captions(self):
        selected_image_indices = self.image_list.get_selected_image_indices()
        selected_image_count = len(selected_image_indices)
        show_alert_when_finished = False
        if selected_image_count > 1:
            confirmation_dialog = CaptionMultipleImagesDialog(
                selected_image_count)
            reply = confirmation_dialog.exec()
            if reply != QMessageBox.StandardButton.Yes:
                return
            show_alert_when_finished = (confirmation_dialog
                                        .show_alert_check_box.isChecked())
        self.set_is_captioning(True)
        caption_settings = self.caption_settings_form.get_caption_settings()
        if caption_settings['caption_position'] != CaptionPosition.DO_NOT_ADD:
            self.image_list_model.add_to_undo_stack(
                action_name=f'Generate '
                            f'{pluralize("Caption", selected_image_count)}',
                should_ask_for_confirmation=selected_image_count > 1)
        if selected_image_count > 1:
            self.progress_bar.setRange(0, selected_image_count)
            self.progress_bar.setValue(0)
            self.progress_bar.show()
        tag_separator = get_tag_separator()
        models_directory_path = self.settings.value(
            'models_directory_path',
            defaultValue=DEFAULT_SETTINGS['models_directory_path'], type=str)
        models_directory_path = (Path(models_directory_path)
                                 if models_directory_path else None)
        self.captioning_thread = CaptioningThread(
            self, self.image_list_model, selected_image_indices,
            caption_settings, tag_separator, models_directory_path)
        self.captioning_thread.text_outputted.connect(
            self.update_console_text_edit)
        self.captioning_thread.clear_console_text_edit_requested.connect(
            self.console_text_edit.clear)
        self.captioning_thread.caption_generated.connect(
            self.caption_generated)
        self.captioning_thread.progress_bar_update_requested.connect(
            self.progress_bar.setValue)
        self.captioning_thread.finished.connect(
            lambda: self.set_is_captioning(False))
        self.captioning_thread.finished.connect(restore_stdout_and_stderr)
        self.captioning_thread.finished.connect(self.progress_bar.hide)
        self.captioning_thread.finished.connect(
            lambda: self.start_cancel_button.setEnabled(True))
        if show_alert_when_finished:
            self.captioning_thread.finished.connect(self.show_alert)
        # Redirect `stdout` and `stderr` so that the outputs are displayed in
        # the console text edit.
        sys.stdout = self.captioning_thread
        sys.stderr = self.captioning_thread
        self.captioning_thread.start()
