import platform
import subprocess
import tempfile

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (QComboBox, QDialog, QFileDialog, QGridLayout,
                               QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit,
                               QPushButton, QTabWidget, QVBoxLayout, QWidget)

from auto_captioning.models_list import MODELS, get_model_class
from auto_captioning.models.wd_tagger import WdTagger
from auto_captioning.models.multi_person_tagger import MultiPersonTagger
from utils.settings import DEFAULT_SETTINGS, get_settings
from utils.settings_widgets import (SettingsBigCheckBox, SettingsComboBox,
                                    SettingsLineEdit, SettingsSpinBox)


# Prompt location constants
PROMPT_LOCATION_PRIMARY = 'Primary Captioning Model'
PROMPT_LOCATION_VLM_PERSON = 'VLM Prompt: Person Description'
PROMPT_LOCATION_VLM_SCENE = 'VLM Prompt: Scene Description'
PROMPT_LOCATION_ENHANCEMENT_LIGHT = 'LLM Prompt: Light Enhancement'
PROMPT_LOCATION_ENHANCEMENT_HEAVY = 'LLM Prompt: Heavy Rewrite'

# Default VLM prompts (for multi-person fine-tune mode)
DEFAULT_VLM_PERSON_PROMPT = """Describe only what you can clearly see of this person in one sentence. Focus on visible clothing, appearance, and pose. Do not infer or imagine details that are not visible."""

DEFAULT_VLM_SCENE_PROMPT = """Describe only the visible setting and background in one brief sentence. Focus on what is actually visible: surfaces, objects, lighting. Do not describe people or infer details you cannot see."""

# Default LLM enhancement prompts
DEFAULT_LIGHT_ENHANCEMENT_PROMPT = """Rewrite this image caption to add the missing details.

Original: {original_desc}
Add these details: {missing_tags}

Rules:
- Maximum 25 words
- Only describe what is visible
- No metaphors or flowery language
- Be factual and direct

Rewritten caption:"""

DEFAULT_HEAVY_REWRITE_PROMPT = """Write a brief image caption using these detected attributes.

Context: {original_desc}
Attributes to include: {all_tags}

Rules:
- Maximum 30 words
- Only describe what is visible
- No metaphors, poetry, or flowery language
- Be factual and direct
- One or two sentences only

Caption:"""


class SettingsDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.settings = get_settings()
        self.setWindowTitle('Settings')
        self.setup_ui()

    @staticmethod
    def get_temp_folder_info() -> tuple[str, str]:
        """Determine system temp folder path and storage type (RAM/disk)."""
        temp_dir = tempfile.gettempdir()
        system = platform.system()

        if system == 'Linux':
            try:
                # Use df -T to check filesystem type
                result = subprocess.run(
                    ['df', '-T', temp_dir],
                    capture_output=True,
                    text=True,
                    timeout=2
                )

                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    if len(lines) >= 2:
                        parts = lines[1].split()
                        if len(parts) >= 2:
                            fs_type = parts[1]
                            # tmpfs = RAM-based
                            if fs_type == 'tmpfs':
                                return temp_dir, 'RAM'
                            else:
                                return temp_dir, f'disk ({fs_type})'
            except Exception:
                pass
            # Fallback for Linux
            return temp_dir, 'disk'

        elif system == 'Windows':
            # Windows temp detection is unreliable without third-party tools
            # Most RAM disks are created by third-party software (ImDisk, etc.)
            # and detection would require complex system queries
            return temp_dir, 'cannot determine'

        elif system == 'Darwin':  # macOS
            # macOS /tmp is typically not tmpfs, but we could check
            # For now, indicate we cannot reliably determine
            return temp_dir, 'cannot determine'

        # Fallback for unknown systems
        return temp_dir, 'unknown'

    def setup_ui(self):
        """Set up the settings dialog UI with tabs."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        # Create tab widget
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)

        # Create tabs
        self.general_tab = QWidget()
        self.prompts_tab = QWidget()

        self.tab_widget.addTab(self.general_tab, 'General')
        self.tab_widget.addTab(self.prompts_tab, 'Prompts')

        # Set up each tab
        self.setup_general_tab()
        self.setup_prompts_tab()

        # Warning label (shared across tabs)
        self.restart_warning = ('Restart the application to apply the new '
                                'settings.')
        self.warning_label = QLabel(self.restart_warning)
        self.warning_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.warning_label.setStyleSheet('color: red;')
        layout.addWidget(self.warning_label)
        self.warning_label.hide()

        # Set minimum size for the dialog
        self.setMinimumSize(700, 500)

    def setup_general_tab(self):
        """Set up the General settings tab."""
        layout = QVBoxLayout(self.general_tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        grid_layout = QGridLayout()
        grid_layout.addWidget(QLabel('Font size (pt)'), 0, 0,
                              Qt.AlignmentFlag.AlignRight)
        grid_layout.addWidget(QLabel('File types to show in image list'), 1, 0,
                              Qt.AlignmentFlag.AlignRight)
        grid_layout.addWidget(QLabel('Image width in image list (px)'), 2, 0,
                              Qt.AlignmentFlag.AlignRight)
        grid_layout.addWidget(QLabel('Tag separator (\\n for newline)'), 3, 0,
                              Qt.AlignmentFlag.AlignRight)
        grid_layout.addWidget(QLabel('Insert space after tag separator'), 4, 0,
                              Qt.AlignmentFlag.AlignRight)
        grid_layout.addWidget(QLabel('Show tag autocomplete suggestions'),
                              5, 0, Qt.AlignmentFlag.AlignRight)
        grid_layout.addWidget(QLabel('Auto-captioning models directory'), 6, 0,
                              Qt.AlignmentFlag.AlignRight)

        # Get system temp folder info for the label
        temp_path, storage_type = self.get_temp_folder_info()
        temp_label = QLabel(f'Multi-person temp file location\n(System: {temp_path} on {storage_type})')
        temp_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        grid_layout.addWidget(temp_label, 8, 0, Qt.AlignmentFlag.AlignRight)

        font_size_spin_box = SettingsSpinBox(
            key='font_size', default=DEFAULT_SETTINGS['font_size'],
            minimum=1, maximum=99)
        font_size_spin_box.valueChanged.connect(self.show_restart_warning)
        # Images that are too small cause lag, so set a minimum width.
        image_list_image_width_spin_box = SettingsSpinBox(
            key='image_list_image_width',
            default=DEFAULT_SETTINGS['image_list_image_width'],
            minimum=16, maximum=9999)
        image_list_image_width_spin_box.valueChanged.connect(
            self.show_restart_warning)
        self.insert_space_after_tag_separator_check_box = SettingsBigCheckBox(
            key='insert_space_after_tag_separator',
            default=DEFAULT_SETTINGS['insert_space_after_tag_separator'])
        self.insert_space_after_tag_separator_check_box.stateChanged.connect(
            self.show_restart_warning)
        tag_separator_line_edit = QLineEdit()
        tag_separator = self.settings.value(
            'tag_separator', defaultValue=DEFAULT_SETTINGS['tag_separator'],
            type=str)
        if tag_separator == '\n':
            tag_separator = r'\n'
            self.disable_insert_space_after_tag_separator_check_box()
        tag_separator_line_edit.setMaximumWidth(50)
        tag_separator_line_edit.setText(tag_separator)
        tag_separator_line_edit.textChanged.connect(
            self.handle_tag_separator_change)
        autocomplete_tags_check_box = SettingsBigCheckBox(
            key='autocomplete_tags',
            default=DEFAULT_SETTINGS['autocomplete_tags'])
        autocomplete_tags_check_box.stateChanged.connect(
            self.show_restart_warning)
        self.models_directory_line_edit = SettingsLineEdit(
            key='models_directory_path',
            default=DEFAULT_SETTINGS['models_directory_path'])
        self.models_directory_line_edit.setMinimumWidth(400)
        self.models_directory_line_edit.setClearButtonEnabled(True)
        self.models_directory_line_edit.textChanged.connect(
            self.show_restart_warning)
        models_directory_button = QPushButton('Select Directory...')
        models_directory_button.setFixedWidth(
            int(models_directory_button.sizeHint().width() * 1.3))
        models_directory_button.clicked.connect(self.set_models_directory_path)
        file_types_line_edit = SettingsLineEdit(
            key='image_list_file_formats',
            default=DEFAULT_SETTINGS['image_list_file_formats'])
        file_types_line_edit.setMinimumWidth(400)
        file_types_line_edit.textChanged.connect(self.show_restart_warning)
        temp_file_location_combo_box = SettingsComboBox(
            key='temp_file_location',
            default=DEFAULT_SETTINGS['temp_file_location'])
        temp_file_location_combo_box.addItems(['Source folder', 'System temp folder'])
        temp_file_location_combo_box.setToolTip(
            'Location for temporary files created during multi-person detection.\n'
            'Source folder: files stored next to images (more private).\n'
            'System temp folder: files stored in system temp directory.')

        grid_layout.addWidget(font_size_spin_box, 0, 1,
                              Qt.AlignmentFlag.AlignLeft)
        grid_layout.addWidget(file_types_line_edit, 1, 1,
                              Qt.AlignmentFlag.AlignLeft)
        grid_layout.addWidget(image_list_image_width_spin_box, 2, 1,
                              Qt.AlignmentFlag.AlignLeft)
        grid_layout.addWidget(tag_separator_line_edit, 3, 1,
                              Qt.AlignmentFlag.AlignLeft)
        grid_layout.addWidget(self.insert_space_after_tag_separator_check_box,
                              4, 1, Qt.AlignmentFlag.AlignLeft)
        grid_layout.addWidget(autocomplete_tags_check_box, 5, 1,
                              Qt.AlignmentFlag.AlignLeft)
        grid_layout.addWidget(self.models_directory_line_edit, 6, 1,
                              Qt.AlignmentFlag.AlignLeft)
        grid_layout.addWidget(models_directory_button, 7, 1,
                              Qt.AlignmentFlag.AlignLeft)
        grid_layout.addWidget(temp_file_location_combo_box, 8, 1,
                              Qt.AlignmentFlag.AlignLeft)
        layout.addLayout(grid_layout)

        # Prevent the grid layout from moving to the center
        layout.addStretch()

    def setup_prompts_tab(self):
        """Set up the Prompts configuration tab."""
        layout = QVBoxLayout(self.prompts_tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(15)

        # Prompt location selector
        location_layout = QHBoxLayout()
        location_layout.addWidget(QLabel('Prompt Location:'))
        self.prompt_location_combo = QComboBox()
        self.prompt_location_combo.addItems([
            PROMPT_LOCATION_PRIMARY,
            PROMPT_LOCATION_VLM_PERSON,
            PROMPT_LOCATION_VLM_SCENE,
            PROMPT_LOCATION_ENHANCEMENT_LIGHT,
            PROMPT_LOCATION_ENHANCEMENT_HEAVY
        ])
        self.prompt_location_combo.currentTextChanged.connect(
            self.on_prompt_location_changed)
        location_layout.addWidget(self.prompt_location_combo, 1)
        layout.addLayout(location_layout)

        # Model selector (only visible for Primary Captioning Model)
        self.model_selector_container = QWidget()
        model_layout = QHBoxLayout(self.model_selector_container)
        model_layout.setContentsMargins(0, 0, 0, 0)
        model_layout.addWidget(QLabel('Model:'))
        self.prompt_model_combo = QComboBox()
        # Populate with VLM models only (exclude taggers)
        self.vlm_models = self._get_vlm_models()
        self.prompt_model_combo.addItems(self.vlm_models)
        self.prompt_model_combo.currentTextChanged.connect(
            self.on_prompt_model_changed)
        model_layout.addWidget(self.prompt_model_combo, 1)
        layout.addWidget(self.model_selector_container)

        # Prompt text editor
        self.prompt_editor = QPlainTextEdit()
        self.prompt_editor.setPlaceholderText('Enter prompt here...')
        self.prompt_editor.textChanged.connect(self.on_prompt_text_changed)
        layout.addWidget(self.prompt_editor, 1)  # stretch factor 1

        # Template variables section
        variables_container = QWidget()
        variables_layout = QVBoxLayout(variables_container)
        variables_layout.setContentsMargins(0, 0, 0, 0)
        variables_layout.setSpacing(5)

        self.variables_label = QLabel('Insert variable:')
        variables_layout.addWidget(self.variables_label)

        # Variable buttons container
        self.variables_buttons_container = QWidget()
        self.variables_buttons_layout = QHBoxLayout(self.variables_buttons_container)
        self.variables_buttons_layout.setContentsMargins(0, 0, 0, 0)
        self.variables_buttons_layout.setSpacing(5)
        variables_layout.addWidget(self.variables_buttons_container)

        layout.addWidget(variables_container)

        # Reset button
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        self.reset_prompt_button = QPushButton('Reset to Default')
        self.reset_prompt_button.clicked.connect(self.reset_current_prompt)
        button_layout.addWidget(self.reset_prompt_button)
        layout.addLayout(button_layout)

        # Track if we're loading to prevent save loops
        self._loading_prompt = False

        # Initialise the UI
        self._update_variables_buttons()
        self._load_current_prompt()

    def _get_vlm_models(self) -> list[str]:
        """Get list of VLM models (excluding taggers)."""
        vlm_models = []
        for model_id in MODELS:
            model_class = get_model_class(model_id)
            if model_class not in (WdTagger, MultiPersonTagger):
                vlm_models.append(model_id)
        return vlm_models

    def _get_prompt_settings_key(self, model_id: str) -> str:
        """Get the settings key for a model's prompt."""
        safe_model_id = model_id.replace('/', '_').replace('\\', '_')
        return f'prompt_{safe_model_id}'

    def _update_variables_buttons(self):
        """Update the variable buttons based on current prompt location."""
        # Clear existing buttons
        while self.variables_buttons_layout.count():
            item = self.variables_buttons_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        location = self.prompt_location_combo.currentText()

        if location == PROMPT_LOCATION_PRIMARY:
            variables = ['{tags}', '{name}', '{directory}']
        elif location in (PROMPT_LOCATION_VLM_PERSON, PROMPT_LOCATION_VLM_SCENE):
            # VLM prompts don't have template variables
            variables = []
        else:
            # LLM enhancement prompts
            variables = ['{original_desc}', '{missing_tags}', '{all_tags}']

        for var in variables:
            btn = QPushButton(var)
            btn.setFixedWidth(120)
            btn.clicked.connect(lambda checked, v=var: self._insert_variable(v))
            self.variables_buttons_layout.addWidget(btn)

        self.variables_buttons_layout.addStretch()

    def _insert_variable(self, variable: str):
        """Insert a variable at the cursor position."""
        cursor = self.prompt_editor.textCursor()
        cursor.insertText(variable)
        self.prompt_editor.setFocus()

    def _load_current_prompt(self):
        """Load the prompt for the current selection."""
        self._loading_prompt = True

        location = self.prompt_location_combo.currentText()

        if location == PROMPT_LOCATION_PRIMARY:
            model_id = self.prompt_model_combo.currentText()
            if model_id:
                key = self._get_prompt_settings_key(model_id)
                saved_prompt = self.settings.value(key, None)

                if saved_prompt is not None:
                    prompt = saved_prompt
                else:
                    # Use model's default prompt
                    model_class = get_model_class(model_id)
                    prompt = model_class.get_default_prompt()

                self.prompt_editor.setPlainText(prompt)

        elif location == PROMPT_LOCATION_VLM_PERSON:
            prompt = self.settings.value(
                'vlm_prompt_person',
                DEFAULT_VLM_PERSON_PROMPT,
                type=str
            )
            self.prompt_editor.setPlainText(prompt)

        elif location == PROMPT_LOCATION_VLM_SCENE:
            prompt = self.settings.value(
                'vlm_prompt_scene',
                DEFAULT_VLM_SCENE_PROMPT,
                type=str
            )
            self.prompt_editor.setPlainText(prompt)

        elif location == PROMPT_LOCATION_ENHANCEMENT_LIGHT:
            prompt = self.settings.value(
                'enhancement_prompt_light',
                DEFAULT_LIGHT_ENHANCEMENT_PROMPT,
                type=str
            )
            self.prompt_editor.setPlainText(prompt)

        elif location == PROMPT_LOCATION_ENHANCEMENT_HEAVY:
            prompt = self.settings.value(
                'enhancement_prompt_heavy',
                DEFAULT_HEAVY_REWRITE_PROMPT,
                type=str
            )
            self.prompt_editor.setPlainText(prompt)

        self._loading_prompt = False

    def _save_current_prompt(self):
        """Save the current prompt to settings."""
        if self._loading_prompt:
            return

        location = self.prompt_location_combo.currentText()
        prompt = self.prompt_editor.toPlainText()

        if location == PROMPT_LOCATION_PRIMARY:
            model_id = self.prompt_model_combo.currentText()
            if model_id:
                key = self._get_prompt_settings_key(model_id)
                self.settings.setValue(key, prompt)

        elif location == PROMPT_LOCATION_VLM_PERSON:
            self.settings.setValue('vlm_prompt_person', prompt)

        elif location == PROMPT_LOCATION_VLM_SCENE:
            self.settings.setValue('vlm_prompt_scene', prompt)

        elif location == PROMPT_LOCATION_ENHANCEMENT_LIGHT:
            self.settings.setValue('enhancement_prompt_light', prompt)

        elif location == PROMPT_LOCATION_ENHANCEMENT_HEAVY:
            self.settings.setValue('enhancement_prompt_heavy', prompt)

    @Slot(str)
    def on_prompt_location_changed(self, location: str):
        """Handle prompt location change."""
        # Show/hide model selector
        is_primary = location == PROMPT_LOCATION_PRIMARY
        self.model_selector_container.setVisible(is_primary)

        # Update variable buttons
        self._update_variables_buttons()

        # Load prompt for new location
        self._load_current_prompt()

    @Slot(str)
    def on_prompt_model_changed(self, model_id: str):
        """Handle model selection change."""
        self._load_current_prompt()

    @Slot()
    def on_prompt_text_changed(self):
        """Handle prompt text changes."""
        self._save_current_prompt()

    @Slot()
    def reset_current_prompt(self):
        """Reset the current prompt to its default."""
        location = self.prompt_location_combo.currentText()

        if location == PROMPT_LOCATION_PRIMARY:
            model_id = self.prompt_model_combo.currentText()
            if model_id:
                model_class = get_model_class(model_id)
                default_prompt = model_class.get_default_prompt()
                self.prompt_editor.setPlainText(default_prompt)

        elif location == PROMPT_LOCATION_VLM_PERSON:
            self.prompt_editor.setPlainText(DEFAULT_VLM_PERSON_PROMPT)

        elif location == PROMPT_LOCATION_VLM_SCENE:
            self.prompt_editor.setPlainText(DEFAULT_VLM_SCENE_PROMPT)

        elif location == PROMPT_LOCATION_ENHANCEMENT_LIGHT:
            self.prompt_editor.setPlainText(DEFAULT_LIGHT_ENHANCEMENT_PROMPT)

        elif location == PROMPT_LOCATION_ENHANCEMENT_HEAVY:
            self.prompt_editor.setPlainText(DEFAULT_HEAVY_REWRITE_PROMPT)

    @Slot()
    def show_restart_warning(self):
        self.warning_label.setText(self.restart_warning)
        self.warning_label.show()

    def disable_insert_space_after_tag_separator_check_box(self):
        self.insert_space_after_tag_separator_check_box.setEnabled(False)
        self.insert_space_after_tag_separator_check_box.setChecked(False)

    @Slot(str)
    def handle_tag_separator_change(self, tag_separator: str):
        if not tag_separator:
            self.warning_label.setText('The tag separator cannot be empty.')
            self.warning_label.show()
            return
        if tag_separator == r'\n':
            tag_separator = '\n'
            self.disable_insert_space_after_tag_separator_check_box()
        else:
            self.insert_space_after_tag_separator_check_box.setEnabled(True)
        self.settings.setValue('tag_separator', tag_separator)
        self.show_restart_warning()

    @Slot()
    def set_models_directory_path(self):
        models_directory_path = self.settings.value(
            'models_directory_path',
            defaultValue=DEFAULT_SETTINGS['models_directory_path'], type=str)
        if models_directory_path:
            initial_directory_path = models_directory_path
        elif self.settings.contains('directory_path'):
            initial_directory_path = self.settings.value('directory_path')
        else:
            initial_directory_path = ''
        models_directory_path = QFileDialog.getExistingDirectory(
            parent=self, caption='Select directory containing auto-captioning '
                                 'models',
            dir=initial_directory_path)
        if models_directory_path:
            self.models_directory_line_edit.setText(models_directory_path)
