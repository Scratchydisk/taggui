import logging
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image as PilImage, ImageDraw
from PySide6.QtCore import QModelIndex, Qt, QTimer, Signal, Slot
from PySide6.QtGui import QBrush, QFontMetrics, QImage, QMovie, QPainter, QPen, QPixmap, QTextCursor, QWheelEvent
from PySide6.QtWidgets import (QAbstractScrollArea, QApplication, QCheckBox,
                               QDialog, QDockWidget, QFormLayout, QFrame,
                               QGraphicsPixmapItem, QGraphicsScene,
                               QGraphicsView, QHBoxLayout, QHeaderView, QLabel,
                               QLineEdit, QMessageBox, QPlainTextEdit,
                               QProgressBar, QPushButton, QScrollArea, QSlider,
                               QTableWidget, QTableWidgetItem, QVBoxLayout,
                               QWidget)

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


logger = logging.getLogger(__name__)


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
        """Handle mouse press for painting, polygon select, or line drawing."""
        if event.button() == Qt.MouseButton.LeftButton and self.detection_dialog:
            scene_pos = self.mapToScene(event.pos())

            # Handle polygon select mode
            if self.detection_dialog.polygon_select_mode:
                self.detection_dialog.add_polygon_point(scene_pos.x(), scene_pos.y())
                return

            # Handle split line mode
            if self.detection_dialog.split_line_mode:
                self.detection_dialog.add_split_line_point(scene_pos.x(), scene_pos.y())
                return

            # Handle painting mode
            if self.edit_mode_enabled:
                # Start painting
                self.is_painting = True
                self.paint_at_position(scene_pos)
                return

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
            # Save masks when done painting
            if self.detection_dialog and self.detection_dialog.pending_mask_save:
                self.detection_dialog.save_edited_masks()
                self.detection_dialog.pending_mask_save = False
        super().mouseReleaseEvent(event)

    def paint_at_position(self, scene_pos):
        """Paint at the given scene position."""
        if not self.detection_dialog:
            return

        # Convert scene coordinates to image coordinates
        # Scene coordinates are in pixels relative to the displayed image
        x = int(scene_pos.x())
        y = int(scene_pos.y())

        # Get brush size from selected person card
        brush_size = 20  # default
        if (self.detection_dialog.selected_card_index is not None and
            self.detection_dialog.selected_card_index < len(self.detection_dialog.person_cards)):
            selected_card = self.detection_dialog.person_cards[self.detection_dialog.selected_card_index]
            brush_size = selected_card.brush_slider.value()

        # Paint on the mask
        self.detection_dialog.paint_mask_at(x, y, brush_size)


class PersonCard(QWidget):
    """Self-contained card widget for a detected person."""

    def __init__(self, person_index, detection, parent_dialog):
        super().__init__()
        self.person_index = person_index
        self.detection = detection
        self.parent_dialog = parent_dialog
        self.is_selected = False
        self.is_expanded = False

        self.setStyleSheet("""
            PersonCard {
                background-color: white;
                border: 2px solid #ddd;
                border-radius: 8px;
            }
            PersonCard[selected="true"] {
                border: 3px solid #2196F3;
                background-color: #e3f2fd;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Header: Person number + select indicator
        header_layout = QHBoxLayout()
        self.person_label = QLabel(f"#{person_index + 1}")
        self.person_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        header_layout.addWidget(self.person_label)
        header_layout.addStretch()
        self.select_indicator = QLabel("")
        self.select_indicator.setStyleSheet("color: #2196F3; font-weight: bold;")
        header_layout.addWidget(self.select_indicator)
        layout.addLayout(header_layout)

        # Thumbnail
        self.thumbnail_label = QLabel()
        self.thumbnail_label.setFixedSize(280, 180)
        self.thumbnail_label.setStyleSheet("border: 1px solid #ccc; background-color: #f9f9f9;")
        self.thumbnail_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumbnail_label.setScaledContents(False)
        layout.addWidget(self.thumbnail_label)

        # Alias input
        alias_layout = QHBoxLayout()
        alias_layout.addWidget(QLabel("Alias:"))
        self.alias_input = QLineEdit()
        self.alias_input.setPlaceholderText("e.g., singer")
        self.alias_input.setText(detection.get('alias', ''))
        self.alias_input.textChanged.connect(self.on_alias_changed)
        alias_layout.addWidget(self.alias_input)
        layout.addLayout(alias_layout)

        # Enabled checkbox
        self.enabled_checkbox = QCheckBox("Enabled for tagging")
        self.enabled_checkbox.setChecked(detection.get('enabled', True))
        self.enabled_checkbox.stateChanged.connect(self.on_enabled_changed)
        layout.addWidget(self.enabled_checkbox)

        # Action buttons
        action_layout = QHBoxLayout()
        self.inverse_button = QPushButton("Inverse")
        self.inverse_button.setToolTip("Create new person from everything except this")
        self.inverse_button.clicked.connect(self.on_inverse_clicked)
        action_layout.addWidget(self.inverse_button)

        self.delete_button = QPushButton("Delete")
        self.delete_button.setStyleSheet("QPushButton { background-color: #f44336; color: white; }")
        self.delete_button.clicked.connect(self.on_delete_clicked)
        action_layout.addWidget(self.delete_button)
        layout.addLayout(action_layout)

        # Edit section (initially hidden, shown when selected)
        self.edit_container = QWidget()
        edit_layout = QVBoxLayout(self.edit_container)
        edit_layout.setContentsMargins(0, 8, 0, 0)
        edit_layout.setSpacing(6)

        # Edit header
        edit_header = QLabel("✏️ Edit Mask")
        edit_header.setStyleSheet("font-weight: bold; color: #333; background-color: #e8f4f8; padding: 4px; border-radius: 3px;")
        edit_layout.addWidget(edit_header)

        # Paint/Erase mode
        mode_layout = QHBoxLayout()
        self.paint_radio = QPushButton("Paint")
        self.paint_radio.setCheckable(True)
        self.paint_radio.setChecked(True)
        self.paint_radio.clicked.connect(lambda: self.set_brush_mode('paint'))
        mode_layout.addWidget(self.paint_radio)

        self.erase_radio = QPushButton("Erase")
        self.erase_radio.setCheckable(True)
        self.erase_radio.clicked.connect(lambda: self.set_brush_mode('erase'))
        mode_layout.addWidget(self.erase_radio)
        edit_layout.addLayout(mode_layout)

        # Brush size
        brush_layout = QHBoxLayout()
        brush_layout.addWidget(QLabel("Size:"))
        self.brush_slider = QSlider(Qt.Orientation.Horizontal)
        self.brush_slider.setMinimum(5)
        self.brush_slider.setMaximum(100)
        self.brush_slider.setValue(20)
        self.brush_slider.valueChanged.connect(self.on_brush_size_changed)
        brush_layout.addWidget(self.brush_slider)
        self.brush_value_label = QLabel("20px")
        brush_layout.addWidget(self.brush_value_label)
        edit_layout.addLayout(brush_layout)

        # Edit tools
        tools_layout = QHBoxLayout()
        self.polygon_button = QPushButton("Polygon")
        self.polygon_button.setToolTip("Draw polygon to add/erase")
        self.polygon_button.clicked.connect(self.on_polygon_clicked)
        tools_layout.addWidget(self.polygon_button)

        self.finish_button = QPushButton("✓ Finish")
        self.finish_button.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
        self.finish_button.clicked.connect(self.on_finish_editing)
        tools_layout.addWidget(self.finish_button)
        edit_layout.addLayout(tools_layout)

        self.edit_container.hide()
        layout.addWidget(self.edit_container)

        # Make card clickable
        self.mousePressEvent = self.on_card_clicked

    def on_card_clicked(self, event):
        """Handle card selection."""
        self.parent_dialog.select_card(self.person_index)

    def set_selected(self, selected):
        """Update visual state when selected/deselected."""
        self.is_selected = selected
        self.setProperty("selected", selected)
        self.style().unpolish(self)
        self.style().polish(self)

        if selected:
            self.select_indicator.setText("★")
            self.edit_container.show()
        else:
            self.select_indicator.setText("")
            self.edit_container.hide()

    def set_thumbnail(self, pixmap):
        """Set the thumbnail image."""
        if pixmap:
            scaled = pixmap.scaled(
                self.thumbnail_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.thumbnail_label.setPixmap(scaled)

    def on_alias_changed(self, text):
        """Handle alias text change."""
        self.detection['alias'] = text
        self.parent_dialog.pending_mask_save = True

    def on_enabled_changed(self, state):
        """Handle enabled checkbox change."""
        enabled = (state == Qt.CheckState.Checked.value)
        self.detection['enabled'] = enabled
        self.parent_dialog.pending_mask_save = True
        self.parent_dialog.redraw_with_highlight()

    def on_inverse_clicked(self):
        """Create inverse crop from this person."""
        self.parent_dialog.create_inverse_crop()

    def on_delete_clicked(self):
        """Delete this person."""
        self.parent_dialog.delete_selected_person()

    def set_brush_mode(self, mode):
        """Set paint or erase mode."""
        self.parent_dialog.brush_mode = mode
        self.paint_radio.setChecked(mode == 'paint')
        self.erase_radio.setChecked(mode == 'erase')

    def on_brush_size_changed(self, value):
        """Handle brush size change."""
        self.brush_value_label.setText(f"{value}px")
        # Brush size is read from selected card's slider in paint_mask_at()

    def on_polygon_clicked(self):
        """Start polygon select mode."""
        self.parent_dialog.start_polygon_select()

    def on_finish_editing(self):
        """Finish editing and save."""
        self.parent_dialog.finish_editing_person()


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

        # Main layout - horizontal split (sidebar | image)
        main_layout = QHBoxLayout(self)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # ========== LEFT SIDEBAR ==========
        sidebar = QWidget()
        sidebar.setMaximumWidth(320)
        sidebar.setMinimumWidth(320)
        sidebar.setStyleSheet("""
            QWidget {
                background-color: #f5f5f5;
            }
        """)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(10, 10, 10, 10)
        sidebar_layout.setSpacing(10)

        # Settings display (collapsible)
        self.settings_label = QLabel()
        self.settings_label.setWordWrap(True)
        self.settings_label.setStyleSheet("font-size: 10px; color: #666; padding: 5px;")
        sidebar_layout.addWidget(self.settings_label)

        # Detection section
        detection_group = QFrame()
        detection_group.setFrameStyle(QFrame.Shape.StyledPanel)
        detection_group.setStyleSheet("QFrame { background-color: white; border-radius: 5px; }")
        detection_layout = QVBoxLayout(detection_group)
        detection_layout.setContentsMargins(10, 10, 10, 10)

        detection_header = QLabel("DETECTION")
        detection_header.setStyleSheet("font-weight: bold; color: #333;")
        detection_layout.addWidget(detection_header)

        # Detection buttons row
        detection_buttons = QHBoxLayout()
        self.reload_button = QPushButton("Reload")
        self.reload_button.setToolTip("Reload saved detections and masks from disk")
        self.reload_button.clicked.connect(self.reload_from_cache)
        self.refresh_button = QPushButton("Re-detect")
        self.refresh_button.setToolTip("Run fresh YOLO person detection and discard any edits")
        self.refresh_button.clicked.connect(self.run_detection)
        detection_buttons.addWidget(self.reload_button)
        detection_buttons.addWidget(self.refresh_button)
        detection_layout.addLayout(detection_buttons)

        # Split merged people checkbox
        self.split_merged_checkbox = QCheckBox("Split Merged People")
        self.split_merged_checkbox.setToolTip(
            "Automatically detect occluded/merged people using WD Tagger + iterative detection.\n"
            "⚠️ Performance: ON = ~6s (automatic), OFF = ~0.4s (manual with Create Inverse)\n"
            "Tip: For best speed, keep OFF and use 'Create Inverse' when needed."
        )
        self.split_merged_checkbox.setChecked(self.detection_settings.get('split_merged_people', True))
        self.split_merged_checkbox.stateChanged.connect(self.on_split_merged_changed)
        detection_layout.addWidget(self.split_merged_checkbox)

        sidebar_layout.addWidget(detection_group)

        # People section (scrollable)
        people_group = QFrame()
        people_group.setFrameStyle(QFrame.Shape.StyledPanel)
        people_group.setStyleSheet("QFrame { background-color: white; border-radius: 5px; }")
        people_layout = QVBoxLayout(people_group)
        people_layout.setContentsMargins(10, 10, 10, 10)

        self.detection_count_label = QLabel("PEOPLE (0 found)")
        self.detection_count_label.setStyleSheet("font-weight: bold; color: #333;")
        people_layout.addWidget(self.detection_count_label)

        # Scrollable person cards area
        self.people_scroll = QScrollArea()
        self.people_scroll.setWidgetResizable(True)
        self.people_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.people_scroll.setStyleSheet("QScrollArea { border: none; background-color: transparent; }")

        self.people_container = QWidget()
        self.people_container.setStyleSheet("background-color: transparent;")
        self.people_layout = QVBoxLayout(self.people_container)
        self.people_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.people_layout.setSpacing(8)
        self.people_scroll.setWidget(self.people_container)
        people_layout.addWidget(self.people_scroll)

        sidebar_layout.addWidget(people_group, 1)  # Take remaining space

        # Tools section
        tools_group = QFrame()
        tools_group.setFrameStyle(QFrame.Shape.StyledPanel)
        tools_group.setStyleSheet("QFrame { background-color: white; border-radius: 5px; }")
        tools_layout = QVBoxLayout(tools_group)
        tools_layout.setContentsMargins(10, 10, 10, 10)

        tools_header = QLabel("TOOLS")
        tools_header.setStyleSheet("font-weight: bold; color: #333;")
        tools_layout.addWidget(tools_header)

        self.add_person_button = QPushButton("+ Add Person")
        self.add_person_button.clicked.connect(self.add_manual_person)
        self.add_person_button.setEnabled(False)
        self.add_person_button.setToolTip("Manually add a person by painting a mask")
        tools_layout.addWidget(self.add_person_button)

        self.split_by_line_button = QPushButton("Split by Line")
        self.split_by_line_button.clicked.connect(self.start_split_by_line)
        self.split_by_line_button.setEnabled(False)
        self.split_by_line_button.setToolTip("Draw a multi-segment line to split image")
        tools_layout.addWidget(self.split_by_line_button)

        self.fit_button = QPushButton("Fit to View")
        # Connection deferred until graphics_view is created
        tools_layout.addWidget(self.fit_button)

        sidebar_layout.addWidget(tools_group)

        # Close button at bottom
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.close)
        sidebar_layout.addWidget(self.close_button)

        main_layout.addWidget(sidebar)

        # ========== RIGHT CONTENT AREA ==========
        content_area = QWidget()
        content_layout = QVBoxLayout(content_area)
        content_layout.setContentsMargins(10, 10, 10, 10)
        content_layout.setSpacing(5)

        # Loading indicator (prominent, centered, overlays content)
        self.loading_label = QLabel("⏳ Running detection, please wait...")
        self.loading_label.setStyleSheet(
            "color: #0066cc; font-weight: bold; font-size: 14px; "
            "background-color: #e6f2ff; padding: 10px; border-radius: 5px;"
        )
        self.loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.loading_label.hide()
        content_layout.addWidget(self.loading_label)

        # Mode banner with action hints
        self.mode_banner = QLabel("🟢 Normal Mode - Select a person to edit")
        self.mode_banner.setStyleSheet("""
            QLabel {
                background-color: #4CAF50;
                color: white;
                padding: 8px;
                font-weight: bold;
                font-size: 13px;
                border-radius: 5px;
            }
        """)
        self.mode_banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        content_layout.addWidget(self.mode_banner)

        # Image display with zoomable graphics view
        self.graphics_view = ZoomableGraphicsView()
        self.graphics_scene = QGraphicsScene()
        self.graphics_view.setScene(self.graphics_scene)
        self.graphics_view.setMinimumHeight(500)
        self.graphics_view.setMinimumWidth(600)
        content_layout.addWidget(self.graphics_view)

        # Now that graphics_view exists, connect fit button
        self.fit_button.clicked.connect(self.graphics_view.reset_zoom)

        # Zoom info
        self.zoom_label = QLabel("Mouse wheel: zoom • Drag: pan")
        self.zoom_label.setStyleSheet("color: #666; font-style: italic; font-size: 11px;")
        self.zoom_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        content_layout.addWidget(self.zoom_label)

        main_layout.addWidget(content_area, 1)  # Take remaining space

        # Finish line button (for split by line mode, overlays on image)
        self.finish_line_button = QPushButton("Finish Line")
        self.finish_line_button.clicked.connect(self.finish_split_line)
        self.finish_line_button.setVisible(False)
        self.finish_line_button.setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-weight: bold; }")
        self.finish_line_button.setParent(content_area)
        self.finish_line_button.move(10, 10)

        # Track selected person
        self.selected_card_index = None
        self.person_cards = []  # List of PersonCard widgets
        self.thumbnail_cache = {}  # Cache thumbnails: key = (index, bbox_tuple) -> QPixmap

        # Initialize editing state
        self.edit_mode_enabled = False
        self.brush_mode = 'paint'  # 'paint' or 'erase'
        self.original_detections = None  # Backup of original masks
        self.is_painting = False
        self.current_display_scale = 1.0  # Scale factor for displayed image vs source
        self.pending_mask_save = False  # Track if we need to save masks

        # Polygon selection state
        self.polygon_select_mode = False
        self.polygon_points = []  # Points for polygon selection
        self.redraw_timer = QTimer(self)
        self.redraw_timer.setSingleShot(True)
        self.redraw_timer.timeout.connect(self._do_redraw)
        self.redraw_pending = False
        self.adding_person_mode = False  # Track if we're adding a new person
        self.new_person_mask = None  # Mask being painted for new person
        self.split_line_mode = False  # Track if we're drawing a split line
        self.split_line_points = []  # Points for the split line (multi-segment)
        self.split_line_preview_side = None  # Which side to preview

        # Preview-scale mask system for performance
        self.preview_scale = 1.0  # Scale factor for preview
        self.preview_size = None  # (width, height) of preview

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts."""
        key = event.key()

        # Esc - Cancel current mode
        if key == Qt.Key.Key_Escape:
            if self.split_line_mode:
                self.cancel_split_by_line()
            elif self.polygon_select_mode:
                self.cancel_polygon_select()
            elif self.adding_person_mode:
                self.adding_person_mode = False
                self.new_person_mask = None
                self.add_person_button.setText("Add Person")
                try:
                    self.add_person_button.clicked.disconnect()
                except:
                    pass
                self.add_person_button.clicked.connect(self.add_manual_person)
                self.update_mode_banner()
            event.accept()
            return

        # Del - Delete selected person
        if key == Qt.Key.Key_Delete:
            if self.selected_card_index is not None:
                self.delete_selected_person()
            event.accept()
            return

        # Ctrl+I - Create inverse
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier and key == Qt.Key.Key_I:
            if self.selected_card_index is not None:
                self.create_inverse_crop()
            event.accept()
            return

        # Number keys 1-9 - Select person
        if Qt.Key.Key_1 <= key <= Qt.Key.Key_9:
            person_index = key - Qt.Key.Key_1
            if 0 <= person_index < len(self.person_cards):
                self.select_card(person_index)
            event.accept()
            return

        # Default handling
        super().keyPressEvent(event)

    def set_image_and_settings(self, image_path: str, detection_settings: dict):
        """Set the image path and detection settings, then load cached or run fresh detection."""
        self.image_path = image_path
        self.detection_settings = detection_settings
        self.update_settings_display()

        # Clear previous results
        self.graphics_scene.clear()

        # Clear previous crop cards
        for card in self.person_cards:
            card.deleteLater()
        self.person_cards.clear()

        # Clear previous crop previews
        while self.crop_previews_layout.count():
            item = self.crop_previews_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Reset selection
        self.selected_card_index = None

        # Hide edit mode if it was enabled
        self.edit_mode_checkbox.setChecked(False)
        self.toggle_edit_mode()

        # Reset mode banner
        self.update_mode_banner()

        # Show loading indicator immediately
        self.loading_label.show()
        self.refresh_button.setEnabled(False)
        self.reload_button.setEnabled(False)

        # Check for cached detections first, otherwise run fresh detection
        QTimer.singleShot(100, self.load_or_run_detection)

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

    def load_or_run_detection(self):
        """Load cached detections if available, otherwise run fresh detection."""
        import time
        start_time = time.time()
        logger.info("=== load_or_run_detection: Checking for cached detections ===")

        # Try to load cached detections first
        load_start = time.time()
        cached_detections = self.load_cached_detections()
        load_time = time.time() - load_start
        logger.info(f"⏱️  load_cached_detections took {load_time:.3f}s")

        if cached_detections is not None:
            logger.info(f"✓ Loaded {len(cached_detections)} cached detections from .masks.npz file - SKIPPING YOLO detection")
            # Update loading text for cache loading
            self.loading_label.setText("📁 Loading cached detections...")
            display_start = time.time()
            self.display_detections(cached_detections, from_cache=True)
            display_time = time.time() - display_start
            logger.info(f"⏱️  display_detections took {display_time:.3f}s")
        else:
            logger.info("✗ No cached detections found, running fresh YOLO detection...")
            # Update loading text for fresh detection
            self.loading_label.setText("⏳ Running detection, please wait...")
            self.run_detection(force_redetect=False)

        total_time = time.time() - start_time
        logger.info(f"⏱️  TOTAL load_or_run_detection took {total_time:.3f}s")

    def on_split_merged_changed(self, state):
        """Handle split merged people checkbox state change."""
        enabled = state == Qt.CheckState.Checked.value
        self.detection_settings['split_merged_people'] = enabled
        logger.info(f"Split merged people setting changed to: {enabled}")
        logger.info(f"Click 'Re-detect' to apply the new setting (detection will be {'slower but automatic' if enabled else 'faster, use Create Inverse for missing people'})")

    def reload_from_cache(self):
        """Manually reload detections and masks from disk cache."""
        logger.info("=== reload_from_cache: Manually reloading from disk ===")

        if not self.image_path:
            return

        # Immediate visual feedback
        self.reload_button.setEnabled(False)
        self.reload_button.setText("Reloading...")
        self.loading_label.setText("📁 Reloading from cache...")
        self.loading_label.show()

        # Clear thumbnail cache to force regeneration with updated masks
        self.thumbnail_cache.clear()
        logger.debug("Cleared thumbnail cache for reload")

        try:
            # Load cached detections
            cached_detections = self.load_cached_detections()

            if cached_detections is not None:
                logger.info(f"✓ Reloaded {len(cached_detections)} cached detections from disk")
                # Display with from_cache=False to clear thumbnail cache
                self.display_detections(cached_detections, from_cache=False)
                QMessageBox.information(
                    self,
                    "Reloaded",
                    f"Successfully reloaded {len(cached_detections)} person(s) from disk cache."
                )
            else:
                logger.warning("No cached detections found to reload")
                QMessageBox.warning(
                    self,
                    "No Cache Found",
                    "No saved detections found for this image. Try running detection first."
                )
        except Exception as e:
            logger.error(f"Failed to reload from cache: {e}")
            QMessageBox.critical(
                self,
                "Reload Error",
                f"Failed to reload from cache: {str(e)}"
            )
        finally:
            # Reset button state
            self.reload_button.setEnabled(True)
            self.reload_button.setText("Reload")
            self.loading_label.hide()

    def load_cached_detections(self):
        """
        Load cached detections from .masks.npz file.

        Returns:
            List of detection dictionaries if file exists, None otherwise
        """
        if not self.image_path:
            return None

        mask_file_path = Path(self.image_path).with_suffix(Path(self.image_path).suffix + '.masks.npz')

        if not mask_file_path.exists():
            return None

        try:
            # Load mask data
            mask_data = np.load(mask_file_path, allow_pickle=True)

            # Determine how many people are saved
            saved_person_count = 0
            for key in mask_data.keys():
                if key.startswith('person_') and key.endswith('_bbox'):
                    person_num = int(key.split('_')[1])
                    saved_person_count = max(saved_person_count, person_num + 1)

            if saved_person_count == 0:
                logger.warning("No people found in cached detection file")
                return None

            # Reconstruct detections from saved data
            detections = []
            for i in range(saved_person_count):
                person_key = f'person_{i}_mask'
                bbox_key = f'person_{i}_bbox'
                enabled_key = f'person_{i}_enabled'
                alias_key = f'person_{i}_alias'

                # Check if this person has required data
                if bbox_key not in mask_data:
                    logger.warning(f"Skipping person {i+1}: missing bbox data")
                    continue

                # Load bbox
                saved_bbox = mask_data[bbox_key]
                x1, y1, x2, y2 = saved_bbox

                # Calculate detection properties
                bbox_width = x2 - x1
                bbox_height = y2 - y1
                area = bbox_width * bbox_height
                center_y = (y1 + y2) // 2

                # Load mask if present
                mask = None
                if person_key in mask_data:
                    mask = mask_data[person_key]

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

                # Create detection dictionary
                detection = {
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': 1.0,  # Cached detections have confidence 1.0
                    'area': int(area),
                    'center_y': int(center_y),
                    'mask': mask,
                    'enabled': enabled,
                    'alias': alias
                }

                detections.append(detection)

            logger.info(f"Successfully loaded {len(detections)} detections from cache")
            return detections if detections else None

        except Exception as e:
            logger.error(f"Failed to load cached detections: {e}")
            return None

    def display_detections(
        self,
        detections: list,
        from_cache: bool = False,
        use_segmentation: bool = False,
        split_merged_people: bool = False,
        original_detection_count: int = None,
        mask_overlapping: bool = False,
        masking_method: str = 'Bounding box'
    ):
        """
        Display detection results in the preview dialog.

        Args:
            detections: List of detection dictionaries
            from_cache: Whether detections are loaded from cache
            use_segmentation: Whether segmentation was used
            split_merged_people: Whether iterative detection was used
            original_detection_count: Original count before iterative detection
            mask_overlapping: Whether overlapping masking is enabled
            masking_method: Method used for masking
        """
        import time
        start_time = time.time()
        try:
            logger.info(f"=== display_detections: Displaying {len(detections)} detections (from_cache={from_cache}) ===")

            # Load image
            logger.debug("Loading image for display...")
            load_img_start = time.time()
            pil_image = PilImage.open(self.image_path).convert('RGB')
            logger.info(f"⏱️  Load image took {time.time() - load_img_start:.3f}s")

            # Store for re-drawing when person selected
            self.current_detections = detections
            self.current_image = pil_image
            self.highlighted_person = None  # Reset highlight

            # Draw bounding boxes and masking visualization
            logger.debug("Drawing detection boundaries on image...")
            draw_start = time.time()
            annotated_image, scale_factor = self.draw_detections(
                pil_image,
                detections,
                self.detection_settings.get('crop_padding', 10),
                mask_overlapping,
                masking_method,
                self.highlighted_person
            )
            logger.info(f"⏱️  draw_detections took {time.time() - draw_start:.3f}s")

            # Store scale factor for coordinate mapping during painting
            self.current_display_scale = scale_factor
            logger.debug(f"Display scale factor: {scale_factor:.3f}")

            # Convert PIL Image to QPixmap in memory (no disk I/O)
            # Use direct numpy conversion instead of PNG serialization for speed
            logger.debug("Converting to QPixmap...")
            convert_start = time.time()

            # Convert PIL to numpy array
            img_array = np.array(annotated_image)
            height, width, channels = img_array.shape
            bytes_per_line = channels * width

            # Create QImage directly from numpy array
            qimage = QImage(img_array.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)

            # QImage doesn't own the data, so we need to keep the numpy array alive
            # Copy the QImage to detach it from the numpy array
            qimage = qimage.copy()

            pixmap = QPixmap.fromImage(qimage)
            logger.info(f"⏱️  Convert to QPixmap took {time.time() - convert_start:.3f}s")

            # Clear and update scene
            logger.debug("Updating scene...")
            scene_start = time.time()
            self.graphics_scene.clear()
            pixmap_item = QGraphicsPixmapItem(pixmap)
            self.graphics_scene.addItem(pixmap_item)
            self.graphics_scene.setSceneRect(pixmap_item.boundingRect())
            logger.info(f"⏱️  Update scene took {time.time() - scene_start:.3f}s")

            # Fit image to view on first load or refresh (delayed to allow layout)
            QTimer.singleShot(50, self.graphics_view.reset_zoom)

            # Update detection count label
            count = len(detections)
            masks_present = any(d.get('mask') is not None for d in detections)

            if from_cache:
                # Indicate these are cached detections
                cache_indicator = "📁 Loaded from cache"
                num_with_masks = sum(1 for d in detections if d.get('mask') is not None)
                mask_info = f" ({num_with_masks} with masks)" if num_with_masks > 0 else ""
                self.detection_count_label.setText(
                    f"{cache_indicator} - {count} {'person' if count == 1 else 'people'}{mask_info}"
                )
            else:
                # Fresh detection
                mask_info = f" (segmentation masks: {'yes' if masks_present else 'no'})" if use_segmentation else ""

                # Show iterative detection info if applied
                iteration_info = ""
                if split_merged_people and original_detection_count is not None and original_detection_count != count:
                    iteration_info = f" [iterative: initial={original_detection_count}]"

                self.detection_count_label.setText(
                    f"{count} {'person' if count == 1 else 'people'} detected{iteration_info}{mask_info}"
                )

            # Update crop cards
            logger.debug(f"Generating crop card thumbnails for {len(detections)} people...")
            cards_start = time.time()
            self.update_crop_cards(detections, from_cache=from_cache)
            logger.info(f"⏱️  update_crop_cards took {time.time() - cards_start:.3f}s")
            logger.debug("Crop card generation complete")

            # Enable mask editing if we have detections with masks
            has_masks = any(d.get('mask') is not None for d in detections)
            self.mask_edit_container.setVisible(has_masks)

            # Enable add person button if image is available
            self.add_person_button.setEnabled(self.current_image is not None)

            # Enable split by line button if we have an image
            self.split_by_line_button.setEnabled(self.current_image is not None)

            # Save masks automatically after fresh detection for faster subsequent loads
            if not from_cache and has_masks:
                save_start = time.time()
                self.save_edited_masks()
                logger.info(f"⏱️  Auto-saved masks to cache in {time.time() - save_start:.3f}s")

            # For cached detections, auto-enable edit mode if masks exist
            if from_cache and has_masks:
                self.edit_mode_checkbox.setChecked(True)
                self.toggle_edit_mode()
                # Re-draw to show loaded masks
                self.redraw_with_highlight()

            total_display_time = time.time() - start_time
            logger.info(f"⏱️  TOTAL display_detections took {total_display_time:.3f}s")

        except Exception as e:
            logger.error(f"Failed to display detections: {e}")
            QMessageBox.critical(
                self,
                "Display Error",
                f"Failed to display detections: {str(e)}"
            )

        finally:
            # Hide loading indicator and re-enable buttons
            self.loading_label.setText("⏳ Running detection, please wait...")  # Reset text
            self.loading_label.hide()
            self.refresh_button.setEnabled(True)
            self.refresh_button.setText("Re-detect")  # Reset button text
            self.reload_button.setEnabled(True)

    def run_detection(self, force_redetect: bool = True):
        """Run person detection and display results.

        Args:
            force_redetect: If True, always run detection. If False, check cache first.
        """
        import time
        logger.info("=== run_detection: Starting YOLO person detection ===")

        if not self.image_path:
            return

        # Immediate visual feedback
        self.refresh_button.setEnabled(False)
        self.refresh_button.setText("Detecting...")
        self.loading_label.setText("⏳ Running detection, please wait...")
        self.loading_label.show()

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
            # Auto-detect device (will use CUDA if available)
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"Creating YOLO detector (model size: {self.detection_settings.get('yolo_model_size', 'm')}, segmentation: {use_segmentation}, device: {device})")
            detector_start = time.time()
            detector = PersonDetector(
                model_size=self.detection_settings.get('yolo_model_size', 'm'),
                device=device,
                conf_threshold=self.detection_settings.get('detection_confidence', 0.5),
                min_size=self.detection_settings.get('detection_min_size', 50),
                max_people=self.detection_settings.get('detection_max_people', 10),
                use_segmentation=use_segmentation
            )
            logger.info(f"⏱️  YOLO model loading took {time.time() - detector_start:.3f}s")

            # Apply split merged people if enabled
            if split_merged_people:
                logger.info("Running iterative detection (split merged people enabled)")
                from auto_captioning.models.multi_person_tagger import MultiPersonTagger
                from auto_captioning.models.wd_tagger import WdTaggerModel
                import numpy as np

                # Load image for WD Tagger
                wd_start = time.time()
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
                logger.info(f"⏱️  WD Tagger preprocessing took {time.time() - wd_start:.3f}s")

                # Use iterative detection if we expect people
                if expected_person_count > 0:
                    logger.info(f"🔍 Running YOLO iterative detection (expecting {expected_person_count} people)...")
                    yolo_start = time.time()
                    detections, original_detection_count = detector.detect_people_iteratively(
                        self.image_path,
                        expected_person_count,
                        max_iterations=3
                    )
                    logger.info(f"⏱️  YOLO iterative detection took {time.time() - yolo_start:.3f}s")
                    logger.info(f"✓ YOLO detection complete: found {len(detections)} people")
                else:
                    # No expected people, use standard detection
                    logger.info("🔍 Running YOLO standard detection...")
                    detections = detector.detect_people(self.image_path)
                    original_detection_count = len(detections)
                    logger.info(f"✓ YOLO detection complete: found {len(detections)} people")
            else:
                # Standard detection (no splitting)
                logger.info("🔍 Running YOLO standard detection...")
                detections = detector.detect_people(self.image_path)
                original_detection_count = len(detections)
                logger.info(f"✓ YOLO detection complete: found {len(detections)} people")

            # Display the detection results
            self.display_detections(
                detections,
                from_cache=False,
                use_segmentation=use_segmentation,
                split_merged_people=split_merged_people,
                original_detection_count=original_detection_count,
                mask_overlapping=mask_overlapping,
                masking_method=masking_method
            )

        except Exception as e:
            # Hide loading indicator and re-enable button on error
            self.loading_label.hide()
            self.refresh_button.setEnabled(True)

            QMessageBox.critical(
                self,
                "Detection Error",
                f"Failed to run detection: {str(e)}"
            )

    def draw_detections(
        self,
        image: PilImage.Image,
        detections: list,
        padding: int,
        mask_overlapping: bool = False,
        masking_method: str = 'Bounding box',
        highlighted_person: int = None
    ) -> tuple[PilImage.Image, float]:
        """Draw bounding boxes, padding, and masking visualization on image.

        Args:
            highlighted_person: If set, only show masking for this person's crop region

        Returns:
            Tuple of (annotated_image, scale_factor) where scale_factor is the ratio
            of display size to original size
        """
        import numpy as np

        # Performance optimization: work at preview resolution for large images
        # This makes boundary drawing and visualization MUCH faster
        max_preview_dimension = 2000  # Max width or height for preview
        original_size = image.size
        scale_factor = 1.0

        if max(original_size) > max_preview_dimension:
            scale_factor = max_preview_dimension / max(original_size)
            new_size = (int(original_size[0] * scale_factor), int(original_size[1] * scale_factor))
            img_copy = image.resize(new_size, PilImage.Resampling.LANCZOS)
            logger.debug(f"Resized image from {original_size} to {new_size} for faster visualization (scale={scale_factor:.2f})")

            # Scale detection data
            scaled_detections = []
            for det in detections:
                scaled_det = det.copy()
                # Scale bbox
                bbox = det['bbox']
                scaled_det['bbox'] = [int(coord * scale_factor) for coord in bbox]
                # Scale mask if present
                if det.get('mask') is not None:
                    mask = det['mask']
                    # Resize mask using PIL for quality
                    mask_img = PilImage.fromarray((mask * 255).astype(np.uint8))
                    mask_resized = mask_img.resize(new_size, PilImage.Resampling.NEAREST)
                    scaled_det['mask'] = np.array(mask_resized) > 127
                else:
                    scaled_det['mask'] = None
                scaled_detections.append(scaled_det)

            detections = scaled_detections
            padding = int(padding * scale_factor)
        else:
            # No scaling needed
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

        return img_copy, scale_factor

    # Obsolete - properties now handled by PersonCard widgets
    def on_alias_changed(self):
        """Handle alias text change (obsolete - now handled by PersonCard)."""
        pass

    def on_enabled_changed_panel(self, state):
        """Handle enabled checkbox change (obsolete - now handled by PersonCard)."""
        pass

    def create_inverse_crop(self):
        """Create a new person from the inverse of the selected person."""
        if self.selected_card_index is None or self.selected_card_index >= len(self.current_detections):
            return

        detection = self.current_detections[self.selected_card_index]
        mask = detection.get('mask')

        if mask is None:
            QMessageBox.warning(
                self,
                "No Mask",
                "Selected person has no segmentation mask. Inverse crop requires a mask."
            )
            return

        # Create inverse mask
        inverse_mask = ~mask

        # Check if there are any pixels
        if not inverse_mask.any():
            QMessageBox.warning(
                self,
                "Empty Mask",
                "The inverse would be empty."
            )
            return

        # Calculate bounding box
        mask_coords = np.argwhere(inverse_mask)
        y_coords = mask_coords[:, 0]
        x_coords = mask_coords[:, 1]

        x1 = int(x_coords.min())
        y1 = int(y_coords.min())
        x2 = int(x_coords.max())
        y2 = int(y_coords.max())

        bbox = [x1, y1, x2, y2]
        width = x2 - x1
        height = y2 - y1
        area = width * height
        center_y = (y1 + y2) // 2

        # Smart alias suggestion
        original_alias = detection.get('alias', '')
        if original_alias:
            suggested_alias = f"{original_alias} inverse"
        else:
            suggested_alias = f"person{self.selected_card_index + 1} inverse"

        # Create new detection
        new_detection = {
            'bbox': bbox,
            'confidence': 1.0,
            'area': area,
            'center_y': center_y,
            'mask': inverse_mask,
            'enabled': True,
            'alias': suggested_alias
        }

        # Add to detections
        self.current_detections.append(new_detection)

        # Update cards
        self.update_crop_cards(self.current_detections)

        # Save
        self.save_edited_masks()

        # Redraw
        self.redraw_with_highlight()

        logger.info(f"Created inverse crop: {suggested_alias}")
        QMessageBox.information(
            self,
            "Person Added",
            f"Inverse person created: '{suggested_alias}'"
        )

    def delete_selected_person(self):
        """Delete the selected person."""
        if self.selected_card_index is None or self.selected_card_index >= len(self.current_detections):
            return

        detection = self.current_detections[self.selected_card_index]
        alias = detection.get('alias', '')
        label = alias if alias else f"Person {self.selected_card_index + 1}"

        # Confirmation
        reply = QMessageBox.question(
            self,
            "Confirm Delete",
            f"Delete '{label}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        # Remove from detections
        del self.current_detections[self.selected_card_index]

        # Update cards
        self.update_crop_cards(self.current_detections)

        # Save
        self.save_edited_masks()

        # Redraw
        self.redraw_with_highlight()

        logger.info(f"Deleted person: {label}")

    def on_person_selected(self):
        """Handle person selection in detection table (deprecated - now using cards)."""
        # This method is kept for backward compatibility but cards use select_card() instead
        pass

        # Show loading indicator
        self.loading_label.setText(f"⏳ Updating preview for Person {row+1}...")
        self.loading_label.show()
        QApplication.processEvents()

        # Set highlighted person and defer redraw to allow UI update
        self.highlighted_person = row

        # Update edit label if in edit mode
        if self.edit_mode_enabled:
            self.selected_person_label.setText(f"Editing: Person {row + 1}")

        QTimer.singleShot(50, self.redraw_with_highlight)

    def schedule_redraw(self):
        """Schedule a redraw with debouncing to avoid excessive redraws during painting."""
        if not self.redraw_timer.isActive():
            # Start timer for 50ms (fast enough to feel responsive, slow enough to batch updates)
            self.redraw_timer.start(50)

    def _do_redraw(self):
        """Internal method called by timer to perform the actual redraw."""
        self.redraw_with_highlight()

    def redraw_with_highlight(self):
        """Redraw the image with the selected person highlighted."""
        if not self.current_image:
            return

        # If adding a person, show the new mask being painted
        if self.adding_person_mode and self.new_person_mask is not None:
            # Create a temporary detection for visualization
            temp_detections = list(self.current_detections) if self.current_detections else []

            # Add temporary detection for the new person being painted
            if self.new_person_mask.any():
                mask_coords = np.argwhere(self.new_person_mask)
                y_coords = mask_coords[:, 0]
                x_coords = mask_coords[:, 1]
                x1 = int(x_coords.min())
                y1 = int(y_coords.min())
                x2 = int(x_coords.max())
                y2 = int(y_coords.max())

                temp_detection = {
                    'bbox': [x1, y1, x2, y2],
                    'confidence': 1.0,
                    'mask': self.new_person_mask
                }
                temp_detections.append(temp_detection)

            # Draw with the temporary detection
            annotated_image, scale_factor = self.draw_detections(
                self.current_image,
                temp_detections,
                self.detection_settings.get('crop_padding', 10),
                False,  # No masking during add mode
                'Segmentation',
                len(temp_detections) - 1 if temp_detections else None  # Highlight the new person
            )
            self.current_display_scale = scale_factor
        elif not self.current_detections:
            # No detections, just show the image
            annotated_image = self.current_image
            self.current_display_scale = 1.0
        else:
            try:
                mask_overlapping = self.detection_settings.get('mask_overlapping_people', True)
                masking_method = self.detection_settings.get('masking_method', 'Bounding box')

                # Draw with highlighted person
                annotated_image, scale_factor = self.draw_detections(
                    self.current_image,
                    self.current_detections,
                    self.detection_settings.get('crop_padding', 10),
                    mask_overlapping,
                    masking_method,
                    self.highlighted_person
                )
                self.current_display_scale = scale_factor
            except Exception as e:
                logger.error(f"Error drawing detections: {e}")
                annotated_image = self.current_image

        try:
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

            # Draw split line if in split line mode
            if self.split_line_mode and self.split_line_points:
                pen = QPen(Qt.GlobalColor.yellow, 3, Qt.PenStyle.DashLine)

                # Draw points
                for px, py in self.split_line_points:
                    self.graphics_scene.addEllipse(px - 5, py - 5, 10, 10, pen)

                # Draw line if we have 2 points
                if len(self.split_line_points) >= 2:
                    x1, y1 = self.split_line_points[0]
                    x2, y2 = self.split_line_points[1]
                    self.graphics_scene.addLine(x1, y1, x2, y2, pen)

            # Draw polygon if in polygon select mode
            if self.polygon_select_mode and self.polygon_points:
                pen = QPen(Qt.GlobalColor.magenta, 3, Qt.PenStyle.SolidLine)
                point_pen = QPen(Qt.GlobalColor.magenta, 2, Qt.PenStyle.SolidLine)
                point_brush = QBrush(Qt.GlobalColor.magenta)

                # Draw points
                for px, py in self.polygon_points:
                    self.graphics_scene.addEllipse(px - 6, py - 6, 12, 12, point_pen, point_brush)

                # Draw lines connecting points
                if len(self.polygon_points) >= 2:
                    for i in range(len(self.polygon_points) - 1):
                        x1, y1 = self.polygon_points[i]
                        x2, y2 = self.polygon_points[i + 1]
                        self.graphics_scene.addLine(x1, y1, x2, y2, pen)

                    # Draw closing line if we have 3+ points (dotted)
                    if len(self.polygon_points) >= 3:
                        closing_pen = QPen(Qt.GlobalColor.magenta, 2, Qt.PenStyle.DashLine)
                        x1, y1 = self.polygon_points[-1]
                        x2, y2 = self.polygon_points[0]
                        self.graphics_scene.addLine(x1, y1, x2, y2, closing_pen)

        finally:
            # Hide loading indicator
            self.loading_label.hide()

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
            self.finish_editing_button.show()
            self.polygon_select_button.show()
            self.reset_masks_button.show()

            # Backup original masks
            if self.original_detections is None:
                import copy
                self.original_detections = copy.deepcopy(self.current_detections)

            # Update selected person label
            if self.highlighted_person is not None:
                self.selected_person_label.setText(f"Editing: Person {self.highlighted_person + 1}")
            else:
                self.selected_person_label.setText("Select a person from the card gallery")

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
            self.finish_editing_button.hide()
            self.polygon_select_button.hide()
            self.finish_polygon_button.hide()
            self.reset_masks_button.hide()

            # Disable painting and restore panning
            self.graphics_view.edit_mode_enabled = False
            self.graphics_view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)

    def finish_editing_person(self):
        """Finish editing the current person - save and update thumbnail."""
        if self.highlighted_person is None or self.highlighted_person >= len(self.current_detections):
            QMessageBox.warning(
                self,
                "No Person Selected",
                "Please select a person to finish editing."
            )
            return

        person_index = self.highlighted_person

        # Save masks to disk
        self.save_edited_masks()

        # Clear thumbnail cache for this person
        self.clear_thumbnail_cache_for_person(person_index)

        # Regenerate the card for this person
        if person_index < len(self.person_cards):
            # Remove old card
            old_card = self.person_cards[person_index]
            self.cards_layout.removeWidget(old_card)
            old_card.deleteLater()

            # Create new card with fresh thumbnail
            detection = self.current_detections[person_index]
            new_card = self._create_crop_card(person_index, detection)
            self.person_cards[person_index] = new_card

            # Insert at correct position
            self.cards_layout.insertWidget(person_index, new_card)

            # Reselect the card
            self.select_card(person_index)

        logger.info(f"Finished editing Person {person_index + 1} - thumbnail updated")

        # Give visual feedback
        self.finish_editing_button.setText("✓ Saved!")
        QTimer.singleShot(1500, lambda: self.finish_editing_button.setText("✓ Finish Editing"))

    def start_polygon_select(self):
        """Start polygon selection mode for adding/erasing mask regions."""
        if self.highlighted_person is None or self.highlighted_person >= len(self.current_detections):
            QMessageBox.warning(
                self,
                "No Person Selected",
                "Please select a person to edit with polygon selection."
            )
            return

        # Cancel any other active modes
        if self.split_line_mode:
            self.cancel_split_by_line()

        self.polygon_select_mode = True
        self.polygon_points = []

        # Update UI with immediate visual feedback
        self.polygon_select_button.setText("Cancel Polygon")
        self.polygon_select_button.setStyleSheet("QPushButton { background-color: #f44336; color: white; }")
        try:
            self.polygon_select_button.clicked.disconnect()
        except:
            pass
        self.polygon_select_button.clicked.connect(self.cancel_polygon_select)
        self.selected_person_label.setText("🟣 POLYGON SELECT MODE - Click points to draw polygon, then 'Finish Polygon'")
        self.selected_person_label.setStyleSheet("background-color: #9C27B0; color: white; padding: 5px; font-weight: bold;")
        self.selected_person_label.show()
        self.finish_polygon_button.setVisible(False)

        # Update mode banner
        self.update_mode_banner()

        logger.info("Entered polygon selection mode")

    def cancel_polygon_select(self):
        """Cancel polygon selection mode."""
        self.polygon_select_mode = False
        self.polygon_points = []

        # Update UI
        self.polygon_select_button.setText("Polygon Select")
        self.polygon_select_button.setStyleSheet("")  # Reset style
        try:
            self.polygon_select_button.clicked.disconnect()
        except:
            pass
        self.polygon_select_button.clicked.connect(self.start_polygon_select)
        if self.highlighted_person is not None:
            self.selected_person_label.setText(f"Editing: Person {self.highlighted_person + 1}")
            self.selected_person_label.setStyleSheet("")  # Reset style
        self.finish_polygon_button.setVisible(False)

        # Update mode banner
        self.update_mode_banner()

        # Redraw to remove polygon
        self.redraw_with_highlight()

        logger.info("Cancelled polygon selection mode")

    def add_polygon_point(self, x: float, y: float):
        """Add a point to the polygon, clamping to image bounds."""
        # Clamp coordinates to displayed image bounds
        # Get scene rect which represents the displayed image dimensions
        scene_rect = self.graphics_scene.sceneRect()
        clamped_x = max(0, min(x, scene_rect.width() - 1))
        clamped_y = max(0, min(y, scene_rect.height() - 1))

        self.polygon_points.append((int(clamped_x), int(clamped_y)))

        if (clamped_x != x or clamped_y != y):
            logger.debug(f"Clamped polygon point from ({x:.0f}, {y:.0f}) to ({clamped_x:.0f}, {clamped_y:.0f})")

        # Update UI
        num_points = len(self.polygon_points)
        if num_points == 1:
            self.selected_person_label.setText("Click more points, then 'Finish Polygon'")
        else:
            self.selected_person_label.setText(f"{num_points} points - Click more or 'Finish Polygon'")

        # Show finish button after 3+ points (minimum for polygon)
        if num_points >= 3:
            self.finish_polygon_button.setVisible(True)

        # Update mode banner
        self.update_mode_banner()

        # Redraw to show the polygon
        self.redraw_with_highlight()

    def finish_polygon_select(self):
        """Finish the polygon and apply to mask."""
        if len(self.polygon_points) < 3:
            QMessageBox.warning(
                self,
                "Not Enough Points",
                "Please add at least 3 points to create a polygon."
            )
            return

        if self.highlighted_person is None or self.highlighted_person >= len(self.current_detections):
            return

        # Ask whether to add or erase
        from PySide6.QtWidgets import QDialog, QVBoxLayout, QRadioButton, QDialogButtonBox, QLabel

        dialog = QDialog(self)
        dialog.setWindowTitle("Apply Polygon")
        layout = QVBoxLayout(dialog)

        label = QLabel("Choose how to apply the polygon:")
        layout.addWidget(label)

        add_radio = QRadioButton("Add to mask")
        add_radio.setChecked(True)
        erase_radio = QRadioButton("Erase from mask")

        layout.addWidget(add_radio)
        layout.addWidget(erase_radio)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        add_to_mask = add_radio.isChecked()

        # Get current person's mask
        detection = self.current_detections[self.highlighted_person]
        mask = detection.get('mask')
        if mask is None:
            QMessageBox.warning(
                self,
                "No Mask",
                "Selected person has no mask to edit."
            )
            return

        # Scale polygon points from display to full-resolution
        scaled_points = []
        for x, y in self.polygon_points:
            if self.current_display_scale != 1.0:
                x = int(x / self.current_display_scale)
                y = int(y / self.current_display_scale)
            scaled_points.append((x, y))

        # Create polygon mask using cv2.fillPoly
        import cv2
        height, width = mask.shape
        polygon_mask = np.zeros((height, width), dtype=np.uint8)
        poly_points = np.array(scaled_points, dtype=np.int32)
        cv2.fillPoly(polygon_mask, [poly_points], 255)
        polygon_mask = polygon_mask > 0

        # Apply to mask
        if add_to_mask:
            mask |= polygon_mask
        else:
            mask &= ~polygon_mask

        detection['mask'] = mask

        # Exit polygon select mode
        self.cancel_polygon_select()

        # Save and update
        self.save_edited_masks()
        self.redraw_with_highlight()

        logger.info(f"Applied polygon ({len(scaled_points)} points) to Person {self.highlighted_person + 1} - {'added' if add_to_mask else 'erased'}")

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
        # Scale coordinates from display space to full-resolution space
        # The displayed image may be scaled down for performance
        if self.current_display_scale != 1.0:
            x = int(x / self.current_display_scale)
            y = int(y / self.current_display_scale)
            brush_size = max(1, int(brush_size / self.current_display_scale))
            logger.debug(f"Scaled paint coords: ({x}, {y}), brush={brush_size} (scale={self.current_display_scale:.3f})")

        # Handle adding new person mode
        if self.adding_person_mode:
            mask = self.new_person_mask
            if mask is None:
                return
        else:
            # Handle editing existing person
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

        # Update the mask
        if self.adding_person_mode:
            self.new_person_mask = mask
        else:
            detection['mask'] = mask

        # Mark that we need to save (but don't save yet - wait for mouse release)
        self.pending_mask_save = True

        # Schedule a redraw (debounced to avoid too many redraws)
        self.schedule_redraw()

    def clear_thumbnail_cache_for_person(self, person_index: int):
        """Clear cached thumbnail for a specific person."""
        if person_index < len(self.current_detections):
            bbox = self.current_detections[person_index]['bbox']
            cache_key = (person_index, tuple(bbox))
            if cache_key in self.thumbnail_cache:
                del self.thumbnail_cache[cache_key]
                logger.debug(f"Cleared thumbnail cache for person {person_index + 1}")

    def save_edited_masks(self):
        """Save edited masks, enabled states, and aliases to a sidecar .masks.npz file."""
        if not self.image_path or not self.current_detections:
            return

        # Clear thumbnail cache for the currently edited person (if in edit mode)
        if self.selected_card_index is not None:
            self.clear_thumbnail_cache_for_person(self.selected_card_index)

        # Create sidecar file path
        mask_file_path = Path(self.image_path).with_suffix(Path(self.image_path).suffix + '.masks.npz')

        # Prepare mask data to save
        mask_data = {}
        for i, detection in enumerate(self.current_detections):
            mask = detection.get('mask')
            if mask is not None:
                # Store mask and bbox for matching
                mask_data[f'person_{i}_mask'] = mask
                mask_data[f'person_{i}_bbox'] = np.array(detection['bbox'])

            # Store enabled state (as int: 1 = True, 0 = False)
            mask_data[f'person_{i}_enabled'] = np.array([1 if detection.get('enabled', True) else 0])

            # Store alias (encode as bytes for numpy)
            alias = detection.get('alias', '')
            mask_data[f'person_{i}_alias'] = np.array([alias], dtype=object)

        # Save to compressed numpy format
        try:
            np.savez_compressed(mask_file_path, **mask_data)
            logger.debug(f"Saved detection data to {mask_file_path}")
        except Exception as e:
            logger.error(f"Failed to save detection data: {e}")

    def load_edited_masks(self):
        """Load edited masks, enabled states, and aliases from sidecar .masks.npz file if it exists."""
        if not self.image_path or not self.current_detections:
            return False

        # Check for sidecar file
        mask_file_path = Path(self.image_path).with_suffix(Path(self.image_path).suffix + '.masks.npz')

        if not mask_file_path.exists():
            return False

        try:
            # Load mask data
            mask_data = np.load(mask_file_path, allow_pickle=True)  # allow_pickle for alias strings

            # Apply masks, enabled states, and aliases to detections by matching bboxes
            for i, detection in enumerate(self.current_detections):
                person_key = f'person_{i}_mask'
                bbox_key = f'person_{i}_bbox'
                enabled_key = f'person_{i}_enabled'
                alias_key = f'person_{i}_alias'

                if person_key in mask_data and bbox_key in mask_data:
                    # Verify bbox matches (detections should be consistent)
                    saved_bbox = mask_data[bbox_key]
                    current_bbox = np.array(detection['bbox'])

                    # Allow small differences due to detection variance
                    if np.allclose(saved_bbox, current_bbox, atol=10):
                        detection['mask'] = mask_data[person_key]
                        logger.debug(f"Loaded edited mask for person {i+1}")
                    else:
                        logger.warning(f"Bbox mismatch for person {i+1}, skipping edited mask")

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

            logger.info(f"Loaded detection data from {mask_file_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load detection data: {e}")
            return False

    def add_manual_person(self):
        """Enter mode to manually add a person by painting."""
        if not self.current_image:
            return

        # Cancel any other active modes
        if self.polygon_select_mode:
            self.cancel_polygon_select()
        if self.split_line_mode:
            self.cancel_split_by_line()

        # Create empty mask for new person
        if self.current_image.height and self.current_image.width:
            self.new_person_mask = np.zeros((self.current_image.height, self.current_image.width), dtype=bool)
        else:
            QMessageBox.warning(
                self,
                "Error",
                "Cannot determine image dimensions."
            )
            return

        # Enter adding person mode
        self.adding_person_mode = True

        # Auto-enable edit mode
        if not self.edit_mode_enabled:
            self.edit_mode_checkbox.setChecked(True)
            self.toggle_edit_mode()

        # Set to paint mode
        self.paint_radio.setChecked(True)
        self.brush_mode = 'paint'

        # Update UI with immediate visual feedback
        self.selected_person_label.setText("🟢 ADD PERSON MODE - Paint the new person, then click 'Finish Adding'")
        self.selected_person_label.setStyleSheet("background-color: #4CAF50; color: white; padding: 5px; font-weight: bold;")
        self.selected_person_label.show()
        self.add_person_button.setText("Finish Adding")
        self.add_person_button.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
        try:
            self.add_person_button.clicked.disconnect()
        except:
            pass
        self.add_person_button.clicked.connect(self.finish_adding_person)

        logger.info("Entered manual person addition mode")

    def finish_adding_person(self):
        """Finish adding a manually painted person."""
        if not self.adding_person_mode or self.new_person_mask is None:
            return

        # Check if mask has any pixels
        if not self.new_person_mask.any():
            QMessageBox.warning(
                self,
                "Empty Mask",
                "Please paint some pixels for the new person before finishing."
            )
            return

        # Calculate bounding box from mask
        mask_coords = np.argwhere(self.new_person_mask)
        if len(mask_coords) == 0:
            QMessageBox.warning(
                self,
                "Empty Mask",
                "The painted mask is empty."
            )
            return

        y_coords = mask_coords[:, 0]
        x_coords = mask_coords[:, 1]

        x1 = int(x_coords.min())
        y1 = int(y_coords.min())
        x2 = int(x_coords.max())
        y2 = int(y_coords.max())

        bbox = [x1, y1, x2, y2]
        width = x2 - x1
        height = y2 - y1
        area = width * height
        center_y = (y1 + y2) // 2

        # Create new detection
        new_detection = {
            'bbox': bbox,
            'confidence': 1.0,  # Manual additions have 100% confidence
            'area': area,
            'center_y': center_y,
            'mask': self.new_person_mask,
            'enabled': True,
            'alias': ''
        }

        # Add to detections
        self.current_detections.append(new_detection)

        # Exit adding person mode
        self.adding_person_mode = False
        self.new_person_mask = None

        # Update UI
        self.add_person_button.setText("Add Person")
        self.add_person_button.setStyleSheet("")  # Reset style
        try:
            self.add_person_button.clicked.disconnect()
        except:
            pass
        self.add_person_button.clicked.connect(self.add_manual_person)
        self.selected_person_label.setStyleSheet("")  # Reset style

        # Update cards
        self.update_crop_cards(self.current_detections)

        # Save to disk
        self.save_edited_masks()

        # Redraw
        self.redraw_with_highlight()

        logger.info(f"Added manual person (bbox: {bbox})")
        QMessageBox.information(
            self,
            "Person Added",
            f"New person added successfully! Total people: {len(self.current_detections)}"
        )

    def start_split_by_line(self):
        """Start split by line mode."""
        if not self.current_image:
            return

        # Cancel any other active modes
        if self.polygon_select_mode:
            self.cancel_polygon_select()

        # Disable edit mode if active
        if self.edit_mode_checkbox.isChecked():
            self.edit_mode_checkbox.setChecked(False)
            self.toggle_edit_mode()

        self.split_line_mode = True
        self.split_line_points = []

        # Update UI with immediate visual feedback
        self.selected_person_label.setText("🟦 SPLIT BY LINE MODE - Click points to draw a line, then 'Finish Line'")
        self.selected_person_label.setStyleSheet("background-color: #2196F3; color: white; padding: 5px; font-weight: bold;")
        self.selected_person_label.show()
        self.split_by_line_button.setText("Cancel Split")
        self.split_by_line_button.setStyleSheet("QPushButton { background-color: #f44336; color: white; }")
        try:
            self.split_by_line_button.clicked.disconnect()
        except:
            pass
        self.split_by_line_button.clicked.connect(self.cancel_split_by_line)

        # Update mode banner
        self.update_mode_banner()

        logger.info("Entered split by line mode")

    def cancel_split_by_line(self):
        """Cancel split by line mode."""
        self.split_line_mode = False
        self.split_line_points = []

        # Update UI
        self.split_by_line_button.setText("Split by Line")
        self.split_by_line_button.setStyleSheet("")  # Reset style
        try:
            self.split_by_line_button.clicked.disconnect()
        except:
            pass
        self.split_by_line_button.clicked.connect(self.start_split_by_line)
        self.selected_person_label.hide()
        self.selected_person_label.setStyleSheet("")  # Reset style
        self.finish_line_button.setVisible(False)

        # Update mode banner
        self.update_mode_banner()

        # Redraw to remove line
        self.redraw_with_highlight()

        logger.info("Cancelled split by line mode")

    def add_split_line_point(self, x: float, y: float):
        """Add a point to the split line, clamping to image bounds."""
        # Clamp coordinates to displayed image bounds
        # Get scene rect which represents the displayed image dimensions
        scene_rect = self.graphics_scene.sceneRect()
        clamped_x = max(0, min(x, scene_rect.width() - 1))
        clamped_y = max(0, min(y, scene_rect.height() - 1))

        self.split_line_points.append((int(clamped_x), int(clamped_y)))

        if (clamped_x != x or clamped_y != y):
            logger.debug(f"Clamped split line point from ({x:.0f}, {y:.0f}) to ({clamped_x:.0f}, {clamped_y:.0f})")

        # Update UI
        num_points = len(self.split_line_points)
        if num_points == 1:
            self.selected_person_label.setText("Click more points, then 'Finish Line'")
        else:
            self.selected_person_label.setText(f"{num_points} points - Click more or 'Finish Line'")

        # Show finish button after 2+ points
        if num_points >= 2:
            self.finish_line_button.setVisible(True)

        # Update mode banner
        self.update_mode_banner()

        # Redraw to show the line
        self.redraw_with_highlight()

    def finish_split_line(self):
        """Finish the line and create two crops, one for each side."""
        if len(self.split_line_points) < 2:
            return

        import cv2
        height, width = self.current_image.height, self.current_image.width

        # Scale split line points from display space to full-resolution space
        scaled_points = []
        for x, y in self.split_line_points:
            if self.current_display_scale != 1.0:
                x = int(x / self.current_display_scale)
                y = int(y / self.current_display_scale)
            scaled_points.append((x, y))

        logger.debug(f"Scaled {len(self.split_line_points)} split line points (scale factor: {self.current_display_scale:.3f})")

        # Use first and last point to define the line
        p1 = np.array(scaled_points[0], dtype=float)
        p2 = np.array(scaled_points[-1], dtype=float)

        # Calculate line direction
        direction = p2 - p1
        if np.linalg.norm(direction) < 1:
            QMessageBox.warning(self, "Invalid Line", "Line is too short.")
            return

        logger.info(f"Split line: p1={p1}, p2={p2}")

        # Create masks using cross product to determine which side of line each pixel is on
        # For a line from p1 to p2, and a point p:
        # cross = (p2.x - p1.x) * (p.y - p1.y) - (p2.y - p1.y) * (p.x - p1.x)
        # cross > 0: left side, cross < 0: right side

        # Create coordinate grids
        y_coords, x_coords = np.mgrid[0:height, 0:width]

        # Calculate cross product for all pixels
        cross = (p2[0] - p1[0]) * (y_coords - p1[1]) - (p2[1] - p1[1]) * (x_coords - p1[0])

        # Create masks
        left_mask = cross > 0
        right_mask = cross < 0

        logger.info(f"Split masks created: left={left_mask.sum()} pixels, right={right_mask.sum()} pixels")

        # Helper function to create detection from mask
        def mask_to_detection(mask: np.ndarray) -> dict:
            """Convert boolean mask to detection dictionary."""
            if not mask.any():
                return None

            # Calculate bbox
            mask_coords = np.argwhere(mask)
            y_coords = mask_coords[:, 0]
            x_coords = mask_coords[:, 1]

            x1, y1 = int(x_coords.min()), int(y_coords.min())
            x2, y2 = int(x_coords.max()), int(y_coords.max())

            return {
                'bbox': [x1, y1, x2, y2],
                'confidence': 1.0,
                'area': (x2 - x1) * (y2 - y1),
                'center_y': (y1 + y2) // 2,
                'mask': mask,
                'enabled': True,
                'alias': ''
            }

        # Create detections for both sides
        new_detections = []
        left_detection = mask_to_detection(left_mask)
        if left_detection:
            new_detections.append(left_detection)

        right_detection = mask_to_detection(right_mask)
        if right_detection:
            new_detections.append(right_detection)

        if not new_detections:
            QMessageBox.warning(self, "Empty Masks", "Failed to create crops from the split line.")
            return

        # Add to detections
        self.current_detections.extend(new_detections)

        # Exit split line mode
        self.cancel_split_by_line()

        # Update UI
        self.update_crop_cards(self.current_detections)
        self.save_edited_masks()
        self.redraw_with_highlight()

        logger.info(f"Created {len(new_detections)} people from split line ({len(self.split_line_points)} points)")
        QMessageBox.information(self, "People Added", f"{len(new_detections)} new people created from split line!")

    def complete_split_by_line(self):
        """Complete the split by line operation."""
        if len(self.split_line_points) != 2:
            return

        p1, p2 = self.split_line_points
        x1, y1 = p1
        x2, y2 = p2

        # Determine if line is more vertical or horizontal
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)

        if dx > dy:
            # More horizontal - split left/right
            options = ["Left side", "Right side"]
            question = "Which side should be the new person?"
        else:
            # More vertical - split top/bottom
            options = ["Top side", "Bottom side"]
            question = "Which side should be the new person?"

        # Ask user which side
        from PySide6.QtWidgets import QInputDialog
        choice, ok = QInputDialog.getItem(
            self,
            "Select Side",
            question,
            options,
            0,
            False
        )

        if not ok:
            self.cancel_split_by_line()
            return

        # Create mask for the selected side
        height, width = self.current_image.height, self.current_image.width
        new_mask = np.zeros((height, width), dtype=bool)

        # Create mask based on which side of the line
        for y in range(height):
            for x in range(width):
                # Calculate which side of the line this pixel is on
                # Using cross product: (p2 - p1) x (point - p1)
                cross = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)

                if dx > dy:
                    # Horizontal line
                    if choice == "Left side":
                        new_mask[y, x] = cross > 0
                    else:  # Right side
                        new_mask[y, x] = cross < 0
                else:
                    # Vertical line
                    if choice == "Top side":
                        new_mask[y, x] = cross > 0
                    else:  # Bottom side
                        new_mask[y, x] = cross < 0

        # Check if mask has any pixels
        if not new_mask.any():
            QMessageBox.warning(
                self,
                "Empty Mask",
                "The selected side has no pixels."
            )
            self.cancel_split_by_line()
            return

        # Calculate bounding box from mask
        mask_coords = np.argwhere(new_mask)
        y_coords = mask_coords[:, 0]
        x_coords = mask_coords[:, 1]

        bbox_x1 = int(x_coords.min())
        bbox_y1 = int(y_coords.min())
        bbox_x2 = int(x_coords.max())
        bbox_y2 = int(y_coords.max())

        bbox = [bbox_x1, bbox_y1, bbox_x2, bbox_y2]
        bbox_width = bbox_x2 - bbox_x1
        bbox_height = bbox_y2 - bbox_y1
        area = bbox_width * bbox_height
        center_y = (bbox_y1 + bbox_y2) // 2

        # Create new detection
        new_detection = {
            'bbox': bbox,
            'confidence': 1.0,
            'area': area,
            'center_y': center_y,
            'mask': new_mask,
            'enabled': True,
            'alias': ''
        }

        # Add to detections
        self.current_detections.append(new_detection)

        # Exit split line mode
        self.cancel_split_by_line()

        # Update table
        self.update_detection_table(self.current_detections)

        # Save to disk
        self.save_edited_masks()

        # Redraw
        self.redraw_with_highlight()

        logger.info(f"Split image by line, created new person (bbox: {bbox})")
        QMessageBox.information(
            self,
            "Person Added",
            f"New person created from {choice.lower()}! Total people: {len(self.current_detections)}"
        )

    def delete_edited_masks(self):
        """Delete the sidecar .masks.npz file."""
        if not self.image_path:
            return

        mask_file_path = Path(self.image_path).with_suffix(Path(self.image_path).suffix + '.masks.npz')

        if mask_file_path.exists():
            try:
                mask_file_path.unlink()
                logger.info(f"Deleted edited masks file: {mask_file_path}")
                QMessageBox.information(
                    self,
                    "Masks Deleted",
                    "Edited masks file has been deleted. Refresh to see original masks."
                )
            except Exception as e:
                logger.error(f"Failed to delete edited masks file: {e}")
                QMessageBox.critical(
                    self,
                    "Error",
                    f"Failed to delete edited masks file: {e}"
                )
        else:
            QMessageBox.information(
                self,
                "No Edited Masks",
                "No edited masks file found for this image."
            )

    def update_crop_cards(self, detections: list, from_cache: bool = False):
        """Update the person card gallery with new PersonCard widgets.

        Args:
            detections: List of detection dictionaries
            from_cache: If True, preserve thumbnail cache. If False, clear it.
        """
        import time
        start_time = time.time()

        # Clear existing cards
        for card in self.person_cards:
            card.deleteLater()
        self.person_cards.clear()

        # Only clear thumbnail cache if running fresh detection
        if not from_cache:
            logger.debug("Clearing thumbnail cache (fresh detection)")
            self.thumbnail_cache.clear()
        else:
            logger.debug("Preserving thumbnail cache (loaded from cache)")

        # Initialize detection properties
        for detection in detections:
            if 'enabled' not in detection:
                detection['enabled'] = True
            if 'alias' not in detection:
                detection['alias'] = ''

        # Generate crop thumbnails and create PersonCard widgets
        for i, detection in enumerate(detections):
            thumb_start = time.time()

            # Create PersonCard widget
            card = PersonCard(i, detection, self)
            self.person_cards.append(card)
            self.people_layout.addWidget(card)

            # Generate and set thumbnail
            try:
                crop_pixmap = self._generate_crop_thumbnail(detection, index=i)
                card.set_thumbnail(crop_pixmap)
                logger.info(f"⏱️  Person {i+1}: generated thumbnail in {time.time() - thumb_start:.3f}s (mask={'yes' if detection.get('mask') is not None else 'no'})")
            except Exception as e:
                logger.error(f"Failed to generate thumbnail for person {i + 1}: {e}")

        logger.info(f"⏱️  update_crop_cards took {time.time() - start_time:.3f}s")

        # Update detection count label
        self.detection_count_label.setText(f"PEOPLE ({len(detections)} found)")

        # Deselect any previously selected card
        self.selected_card_index = None

    def _generate_crop_thumbnail(self, detection: dict, index: int = None) -> QPixmap:
        """Generate a crop thumbnail for a detection (with caching)."""
        import time
        start_time = time.time()

        if not self.current_image:
            return QPixmap()

        # Check cache first (keyed by index and bbox)
        bbox = detection['bbox']
        if index is not None:
            cache_key = (index, tuple(bbox))
            if cache_key in self.thumbnail_cache:
                logger.debug(f"⏱️  Person {index+1}: thumbnail from cache (0.000s)")
                return self.thumbnail_cache[cache_key]

        # Use segmentation-based extraction
        mask = detection.get('mask')

        if mask is not None:
            # Extract segmented person
            from auto_captioning.models.multi_person_tagger import MultiPersonTagger
            # Create a temporary tagger instance for the extraction method
            temp_settings = {
                'mask_overlapping_people': False,
                'masking_method': 'Segmentation',
                'preserve_target_bbox': False,
                'crop_padding': 10,
                'mask_erosion_size': 0,
                'mask_dilation_size': 0,
                'mask_blur_size': 0
            }

            # Create temporary instance (we just need the extraction method)
            img_array = np.array(self.current_image)
            num_channels = img_array.shape[2] if len(img_array.shape) == 3 else 1
            white_value = [255] * num_channels if num_channels > 1 else 255
            result_array = np.full_like(img_array, white_value)
            result_array[mask] = img_array[mask]
            result_image = PilImage.fromarray(result_array)

            # Find tight bounds and crop
            mask_coords = np.argwhere(mask)
            if len(mask_coords) > 0:
                y_coords = mask_coords[:, 0]
                x_coords = mask_coords[:, 1]
                y_min, y_max = y_coords.min(), y_coords.max()
                x_min, x_max = x_coords.min(), x_coords.max()

                padding = 10
                crop_x1 = max(0, x_min - padding)
                crop_y1 = max(0, y_min - padding)
                crop_x2 = min(self.current_image.width, x_max + padding + 1)
                crop_y2 = min(self.current_image.height, y_max + padding + 1)

                cropped = result_image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
            else:
                # Fallback to bbox
                x1, y1, x2, y2 = bbox
                cropped = self.current_image.crop((x1, y1, x2, y2))
        else:
            # Use bbox
            x1, y1, x2, y2 = bbox
            cropped = self.current_image.crop((x1, y1, x2, y2))

        # Convert to QPixmap
        buffer = BytesIO()
        cropped.save(buffer, format='PNG')
        buffer.seek(0)
        qimage = QImage.fromData(buffer.getvalue())
        pixmap = QPixmap.fromImage(qimage)

        # Store in cache
        if index is not None:
            cache_key = (index, tuple(bbox))
            self.thumbnail_cache[cache_key] = pixmap

        elapsed = time.time() - start_time
        logger.info(f"⏱️  Person {index+1 if index is not None else '?'}: generated thumbnail in {elapsed:.3f}s (mask={'yes' if mask is not None else 'no'})")

        return pixmap

    def select_card(self, index: int):
        """Select a person card."""
        if index == self.selected_card_index:
            return  # Already selected

        # Deselect previous card
        if self.selected_card_index is not None and self.selected_card_index < len(self.person_cards):
            self.person_cards[self.selected_card_index].set_selected(False)

        # Update selection
        self.selected_card_index = index

        # Select new card (PersonCard will handle visual updates and show/hide edit controls)
        if index < len(self.person_cards):
            self.person_cards[index].set_selected(True)

        # Highlight selected person on image
        self.highlighted_person = index
        self.redraw_with_highlight()

        # Update mode banner
        self.update_mode_banner()

    # Obsolete - properties now in PersonCard widgets
    def update_properties_panel(self):
        """Update the properties panel (obsolete - now in PersonCard)."""
        pass

    def update_status_footer(self):
        """Update the status footer (obsolete - count now in detection_count_label)."""
        pass

    def update_mode_banner(self):
        """Update the mode banner based on current state."""
        if self.polygon_select_mode:
            points_count = len(self.polygon_points)
            if points_count == 0:
                text = "🟣 Polygon Select Mode - Click to add points (min 3)"
            elif points_count < 3:
                text = f"🟣 Polygon Select Mode - {points_count} points. Need {3 - points_count} more"
            else:
                text = f"🟣 Polygon Select Mode - {points_count} points. Click more or 'Finish Polygon'"
            style = "background-color: #9C27B0;"  # Purple
        elif self.split_line_mode:
            points_count = len(self.split_line_points)
            if points_count == 0:
                text = "🟡 Split Line Mode - Click to add points"
            elif points_count == 1:
                text = "🟡 Split Line Mode - Click more points, then 'Finish Line'"
            else:
                text = f"🟡 Split Line Mode - {points_count} points. Click more or 'Finish Line'"
            style = "background-color: #FFC107;"  # Yellow
        elif self.adding_person_mode:
            text = "🟠 Add Person Mode - Paint to create mask, then 'Finish Adding'"
            style = "background-color: #FF9800;"  # Orange
        elif self.edit_mode_enabled and self.selected_card_index is not None:
            alias = self.current_detections[self.selected_card_index].get('alias', '')
            label = alias if alias else f"Person {self.selected_card_index + 1}"
            text = f"🔵 Edit Mode: {label} - Paint or erase mask"
            style = "background-color: #2196F3;"  # Blue
        elif self.selected_card_index is not None:
            alias = self.current_detections[self.selected_card_index].get('alias', '')
            label = alias if alias else f"Person {self.selected_card_index + 1}"
            text = f"📝 Selected: {label} - Edit properties below or enable Edit Mode"
            style = "background-color: #4CAF50;"  # Green
        else:
            text = "🟢 Normal Mode - Select a person card below to edit"
            style = "background-color: #4CAF50;"  # Green

        self.mode_banner.setText(text)
        self.mode_banner.setStyleSheet(f"""
            QLabel {{
                {style}
                color: white;
                padding: 10px;
                font-weight: bold;
                font-size: 14px;
                border-radius: 5px;
            }}
        """)

    # Keep old method name for backwards compatibility during refactor
    def update_detection_table(self, detections: list):
        """Update the detection info table."""
        # Block signals while updating to avoid triggering itemChanged
        self.detection_table.blockSignals(True)

        self.detection_table.setRowCount(len(detections))

        for i, detection in enumerate(detections):
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            confidence = detection['confidence']

            # Initialize enabled state if not present (default to True)
            if 'enabled' not in detection:
                detection['enabled'] = True

            # Initialize alias if not present (default to empty)
            if 'alias' not in detection:
                detection['alias'] = ''

            # Column 0: Person number
            self.detection_table.setItem(i, 0, QTableWidgetItem(str(i + 1)))

            # Column 1: Enabled checkbox
            checkbox = QCheckBox()
            checkbox.setChecked(detection['enabled'])
            checkbox.setStyleSheet("margin-left: 50%; margin-right: 50%;")
            checkbox_container = QWidget()
            checkbox_layout = QHBoxLayout(checkbox_container)
            checkbox_layout.addWidget(checkbox)
            checkbox_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            checkbox_layout.setContentsMargins(0, 0, 0, 0)
            self.detection_table.setCellWidget(i, 1, checkbox_container)
            # Connect checkbox to update detection state
            checkbox.stateChanged.connect(
                lambda state, idx=i: self.on_person_enabled_changed(idx, state == Qt.CheckState.Checked.value)
            )

            # Column 2: Alias (editable)
            alias_item = QTableWidgetItem(detection['alias'])
            alias_item.setFlags(alias_item.flags() | Qt.ItemFlag.ItemIsEditable)
            self.detection_table.setItem(i, 2, alias_item)

            # Column 3: Confidence
            confidence_item = QTableWidgetItem(f"{confidence:.3f}")
            confidence_item.setFlags(confidence_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.detection_table.setItem(i, 3, confidence_item)

            # Column 4: Size
            size_item = QTableWidgetItem(f"{width}x{height}")
            size_item.setFlags(size_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.detection_table.setItem(i, 4, size_item)

            # Column 5: Bbox coordinates
            bbox_item = QTableWidgetItem(f"({x1},{y1},{x2},{y2})")
            bbox_item.setFlags(bbox_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.detection_table.setItem(i, 5, bbox_item)

        # Re-enable signals
        self.detection_table.blockSignals(False)

    def on_person_enabled_changed(self, person_idx: int, enabled: bool):
        """Handle person enabled/disabled state change."""
        if 0 <= person_idx < len(self.current_detections):
            self.current_detections[person_idx]['enabled'] = enabled
            # Save to sidecar file
            self.save_edited_masks()
            logger.info(f"Person {person_idx + 1} {'enabled' if enabled else 'disabled'}")

    def on_detection_table_changed(self, item: QTableWidgetItem):
        """Handle changes to table items (e.g., alias editing)."""
        if item.column() == 2:  # Alias column
            row = item.row()
            if 0 <= row < len(self.current_detections):
                new_alias = item.text().strip()
                self.current_detections[row]['alias'] = new_alias
                # Save to sidecar file
                self.save_edited_masks()
                logger.info(f"Person {row + 1} alias set to: '{new_alias}'")

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

            # Generate crops for each person (skip disabled ones)
            for i, detection in enumerate(self.current_detections):
                # Skip if person is disabled
                if not detection.get('enabled', True):
                    continue

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
