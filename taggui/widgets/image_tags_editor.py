from PySide6.QtCore import (QItemSelectionModel, QModelIndex, QStringListModel,
                            QTimer, Qt, Signal, Slot)
from PySide6.QtGui import QKeyEvent, QCursor
from PySide6.QtWidgets import (QAbstractItemView, QCompleter, QDockWidget,
                               QHBoxLayout, QLabel, QLineEdit, QListView, QMessageBox,
                               QPlainTextEdit, QVBoxLayout, QWidget)
from transformers import PreTrainedTokenizerBase

from models.proxy_image_list_model import ProxyImageListModel
from models.tag_counter_model import TagCounterModel
from utils.image import Image
from utils.settings import DEFAULT_SETTINGS, get_settings
from utils.text_edit_item_delegate import TextEditItemDelegate
from utils.utils import get_confirmation_dialog_reply
from widgets.image_list import ImageList

MAX_TOKEN_COUNT = 75


class TagInputBox(QLineEdit):
    tags_addition_requested = Signal(list, list)

    def __init__(self, image_tag_list_model: QStringListModel,
                 tag_counter_model: TagCounterModel, image_list: ImageList,
                 tag_separator: str):
        super().__init__()
        self.image_tag_list_model = image_tag_list_model
        self.image_list = image_list
        self.tag_separator = tag_separator

        self.setPlaceholderText('Add Tag')
        self.setStyleSheet('padding: 8px;')
        settings = get_settings()
        autocomplete_tags = settings.value(
            'autocomplete_tags',
            defaultValue=DEFAULT_SETTINGS['autocomplete_tags'], type=bool)
        if autocomplete_tags:
            self.completer = QCompleter(tag_counter_model)
            self.setCompleter(self.completer)
            self.completer.activated.connect(lambda text: self.add_tag(text))
            # Clear the input box after the completer inserts the tag into it.
            self.completer.activated.connect(
                lambda: QTimer.singleShot(0, self.clear))
        else:
            self.completer = None

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() not in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            super().keyPressEvent(event)
            return
        # If Ctrl+Enter is pressed and the completer is visible, add the first
        # tag in the completer popup.
        if (event.modifiers() == Qt.KeyboardModifier.ControlModifier
                and self.completer is not None
                and self.completer.popup().isVisible()):
            first_tag = self.completer.popup().model().data(
                self.completer.model().index(0, 0), Qt.ItemDataRole.EditRole)
            self.add_tag(first_tag)
        # Otherwise, add the tag in the input box.
        else:
            self.add_tag(self.text())
        self.clear()
        if self.completer is not None:
            self.completer.popup().hide()

    def add_tag(self, tag: str):
        if not tag:
            return
        tags = tag.split(self.tag_separator)
        selected_image_indices = self.image_list.get_selected_image_indices()
        selected_image_count = len(selected_image_indices)
        if len(tags) == 1 and selected_image_count == 1:
            # Add an empty tag and set it to the new tag.
            self.image_tag_list_model.insertRow(
                self.image_tag_list_model.rowCount())
            new_tag_index = self.image_tag_list_model.index(
                self.image_tag_list_model.rowCount() - 1)
            self.image_tag_list_model.setData(new_tag_index, tag)
            return
        if selected_image_count > 1:
            if len(tags) > 1:
                question = (f'Add tags to {selected_image_count} selected '
                            f'images?')
            else:
                question = (f'Add tag "{tags[0]}" to {selected_image_count} '
                            f'selected images?')
            reply = get_confirmation_dialog_reply(title='Add Tag',
                                                  question=question)
            if reply != QMessageBox.StandardButton.Yes:
                return
        self.tags_addition_requested.emit(tags, selected_image_indices)


class ImageTagsList(QListView):
    def __init__(self, image_tag_list_model: QStringListModel):
        super().__init__()
        self.image_tag_list_model = image_tag_list_model
        self.setModel(self.image_tag_list_model)
        self.setItemDelegate(TextEditItemDelegate(self))
        self.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setWordWrap(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)

    def keyPressEvent(self, event: QKeyEvent):
        """
        Delete selected tags when the delete key or backspace key is pressed.
        """
        if event.key() not in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            super().keyPressEvent(event)
            return
        rows_to_remove = [index.row() for index in self.selectedIndexes()]
        if not rows_to_remove:
            return
        remaining_tags = [tag for i, tag
                          in enumerate(self.image_tag_list_model.stringList())
                          if i not in rows_to_remove]
        self.image_tag_list_model.setStringList(remaining_tags)
        min_removed_row = min(rows_to_remove)
        remaining_row_count = self.image_tag_list_model.rowCount()
        if min_removed_row < remaining_row_count:
            self.select_tag(min_removed_row)
        elif remaining_row_count:
            # Select the last tag.
            self.select_tag(remaining_row_count - 1)

    def select_tag(self, row: int):
        # If the current index is not set, using the arrow keys to navigate
        # through the tags after selecting the tag will not work.
        self.setCurrentIndex(self.image_tag_list_model.index(row))
        self.selectionModel().select(
            self.image_tag_list_model.index(row),
            QItemSelectionModel.SelectionFlag.ClearAndSelect)


class ImageTagsEditor(QDockWidget):
    # Signal emitted when caption is edited by user
    caption_changed = Signal(QModelIndex, str)

    def __init__(self, proxy_image_list_model: ProxyImageListModel,
                 tag_counter_model: TagCounterModel,
                 image_tag_list_model: QStringListModel, image_list: ImageList,
                 tokenizer: PreTrainedTokenizerBase, tag_separator: str):
        super().__init__()
        self.proxy_image_list_model = proxy_image_list_model
        self.image_tag_list_model = image_tag_list_model
        self.tokenizer = tokenizer
        self.tag_separator = tag_separator
        self.image_index = None
        self.current_caption = None
        # Flag to prevent saving when loading a new image
        self._loading_caption = False
        # Check if we should start in caption mode
        settings = get_settings()
        show_captions = settings.value(
            'show_captions_mode', defaultValue=False, type=bool)
        self.view_mode = 'caption' if show_captions else 'tags'

        # Each `QDockWidget` needs a unique object name for saving its state.
        self.setObjectName('image_tags_editor')
        self.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea
                             | Qt.DockWidgetArea.RightDockWidgetArea)
        self.tag_input_box = TagInputBox(self.image_tag_list_model,
                                         tag_counter_model, image_list,
                                         tag_separator)
        self.image_tags_list = ImageTagsList(self.image_tag_list_model)

        # Caption editor
        self.caption_text_edit = QPlainTextEdit()
        self.caption_text_edit.setPlaceholderText('No caption available')
        self.caption_text_edit.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.caption_text_edit.setMinimumHeight(100)
        self.caption_text_edit.textChanged.connect(self._on_caption_text_changed)

        # Set initial visibility based on view mode
        if self.view_mode == 'caption':
            self.setWindowTitle('Image Caption')
            self.tag_input_box.hide()
            self.image_tags_list.hide()
            self.caption_text_edit.show()
        else:
            self.setWindowTitle('Image Tags')
            self.caption_text_edit.hide()

        self.token_count_label = QLabel()

        # Clickable file source labels
        self.file_source_container = QWidget()
        file_source_layout = QHBoxLayout(self.file_source_container)
        file_source_layout.setContentsMargins(0, 0, 0, 0)
        file_source_layout.setSpacing(5)

        self.view_tags_label = QLabel()
        self.view_tags_label.setStyleSheet('color: grey; font-size: 10px; text-decoration: underline;')
        self.view_tags_label.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.view_tags_label.mousePressEvent = lambda event: self.switch_to_tags_view()

        self.separator_label = QLabel('|')
        self.separator_label.setStyleSheet('color: grey; font-size: 10px;')

        self.view_caption_label = QLabel()
        self.view_caption_label.setStyleSheet('color: grey; font-size: 10px; text-decoration: underline;')
        self.view_caption_label.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.view_caption_label.mousePressEvent = lambda event: self.switch_to_caption_view()

        file_source_layout.addWidget(self.view_tags_label)
        file_source_layout.addWidget(self.separator_label)
        file_source_layout.addWidget(self.view_caption_label)
        file_source_layout.addStretch()

        # A container widget is required to use a layout with a `QDockWidget`.
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(self.tag_input_box)
        layout.addWidget(self.image_tags_list)
        layout.addWidget(self.caption_text_edit)
        layout.addWidget(self.token_count_label)
        layout.addWidget(self.file_source_container)
        self.setWidget(container)

        # When a tag is added, select it and scroll to the bottom of the list.
        self.image_tag_list_model.rowsInserted.connect(
            lambda _, __, last_index:
            self.image_tags_list.selectionModel().select(
                self.image_tag_list_model.index(last_index),
                QItemSelectionModel.SelectionFlag.ClearAndSelect))
        self.image_tag_list_model.rowsInserted.connect(
            self.image_tags_list.scrollToBottom)
        # `rowsInserted` does not have to be connected because `dataChanged`
        # is emitted when a tag is added.
        self.image_tag_list_model.modelReset.connect(self.count_tokens)
        self.image_tag_list_model.dataChanged.connect(self.count_tokens)

    @Slot()
    def count_tokens(self):
        caption = self.tag_separator.join(
            self.image_tag_list_model.stringList())
        # Subtract 2 for the `<|startoftext|>` and `<|endoftext|>` tokens.
        caption_token_count = len(self.tokenizer(caption).input_ids) - 2
        if caption_token_count > MAX_TOKEN_COUNT:
            self.token_count_label.setStyleSheet('color: red;')
        else:
            self.token_count_label.setStyleSheet('')
        self.token_count_label.setText(f'{caption_token_count} / '
                                       f'{MAX_TOKEN_COUNT} Tokens')

    @Slot()
    def select_first_tag(self):
        if self.image_tag_list_model.rowCount() == 0:
            return
        self.image_tags_list.select_tag(0)

    def select_last_tag(self):
        tag_count = self.image_tag_list_model.rowCount()
        if tag_count == 0:
            return
        self.image_tags_list.select_tag(tag_count - 1)

    @Slot()
    def load_image_tags(self, proxy_image_index: QModelIndex):
        self.image_index = self.proxy_image_list_model.mapToSource(
            proxy_image_index)
        image: Image = self.proxy_image_list_model.data(
            proxy_image_index, Qt.ItemDataRole.UserRole)
        # If the string list already contains the image's tags AND caption hasn't
        # changed, do not reload. This is the case when the tags are edited directly
        # through the image tags editor. Removing this check breaks the functionality
        # of reordering multiple tags at the same time because it gets interrupted
        # after one tag is moved.
        current_string_list = self.image_tag_list_model.stringList()
        old_caption = getattr(self, 'current_caption', None)
        # Also check if caption UI state needs updating (e.g., caption label visibility, text content)
        caption_changed = old_caption != image.caption
        caption_ui_needs_update = (
            (image.caption and not self.view_caption_label.isVisible()) or
            (not image.caption and self.view_caption_label.isVisible()) or
            (self.caption_text_edit.toPlainText() != (image.caption or ''))
        )
        if current_string_list == image.tags and not caption_changed and not caption_ui_needs_update:
            return
        self.image_tag_list_model.setStringList(image.tags)
        self.count_tokens()

        # Store caption for view switching
        self.current_caption = image.caption

        # Update clickable labels
        if image.tags_file_type == '.tags.txt':
            tags_text = f"View tags ({image.path.name}.tags.txt)"
        else:
            tags_text = f"View tags ({image.path.stem}.txt)"

        if image.caption:
            caption_text = f"View caption ({image.path.name}.caption.txt)"
            self.view_caption_label.setText(caption_text)
            self.view_caption_label.show()
            self.separator_label.show()
        else:
            self.view_caption_label.hide()
            self.separator_label.hide()

        self.view_tags_label.setText(tags_text)

        # Update styling to indicate active view
        self._update_view_styling()

        # Load caption into text edit (use flag to prevent auto-save triggering)
        # Only update if content actually changed to avoid disrupting user's typing
        current_text = self.caption_text_edit.toPlainText()
        new_text = image.caption or ''
        if current_text != new_text:
            self._loading_caption = True
            self.caption_text_edit.setPlainText(new_text)
            self._loading_caption = False

        if self.image_tags_list.hasFocus():
            self.select_first_tag()

    def switch_to_tags_view(self):
        """Switch to viewing tags."""
        if self.view_mode == 'tags':
            return
        self.view_mode = 'tags'
        self.setWindowTitle('Image Tags')
        self.tag_input_box.show()
        self.image_tags_list.show()
        self.caption_text_edit.hide()
        self._update_view_styling()

    def switch_to_caption_view(self, force: bool = False):
        """Switch to viewing caption.

        Args:
            force: If True, switch even if no caption available (for global toggle)
        """
        if self.view_mode == 'caption':
            return
        if not force and not self.current_caption:
            return  # Don't switch if no caption available (when clicking label)
        self.view_mode = 'caption'
        self.setWindowTitle('Image Caption')
        self.tag_input_box.hide()
        self.image_tags_list.hide()
        self.caption_text_edit.show()
        self._update_view_styling()

    def _update_view_styling(self):
        """Update label styling to indicate which view is active."""
        if self.view_mode == 'tags':
            self.view_tags_label.setStyleSheet(
                'color: blue; font-size: 10px; font-weight: bold; text-decoration: none;'
            )
            self.view_caption_label.setStyleSheet(
                'color: grey; font-size: 10px; text-decoration: underline;'
            )
        else:
            self.view_tags_label.setStyleSheet(
                'color: grey; font-size: 10px; text-decoration: underline;'
            )
            self.view_caption_label.setStyleSheet(
                'color: blue; font-size: 10px; font-weight: bold; text-decoration: none;'
            )

    @Slot()
    def _on_caption_text_changed(self):
        """Handle caption text changes and emit signal to save."""
        # Don't save when we're just loading a new image
        if self._loading_caption:
            return
        # Don't save if no image is selected
        if self.image_index is None:
            return
        new_caption = self.caption_text_edit.toPlainText()
        # Update label visibility if caption state changed (empty <-> non-empty)
        had_caption = bool(self.current_caption)
        has_caption = bool(new_caption)
        if had_caption != has_caption:
            if has_caption:
                self.view_caption_label.show()
                self.separator_label.show()
            else:
                self.view_caption_label.hide()
                self.separator_label.hide()
        # Update local cache and emit signal to save
        self.current_caption = new_caption
        self.caption_changed.emit(self.image_index, new_caption)

    @Slot()
    def reload_image_tags_if_changed(self, first_changed_index: QModelIndex,
                                     last_changed_index: QModelIndex):
        """
        Reload the tags for the current image if its index is in the range of
        changed indices.
        """
        if (first_changed_index.row() <= self.image_index.row()
                <= last_changed_index.row()):
            proxy_image_index = self.proxy_image_list_model.mapFromSource(
                self.image_index)
            self.load_image_tags(proxy_image_index)
