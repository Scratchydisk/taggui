from PySide6.QtWidgets import QCheckBox, QPushButton

from utils.settings import DEFAULT_SETTINGS, get_settings


class BigPushButton(QPushButton):
    def __init__(self, text: str):
        super().__init__(text)
        new_size = self.sizeHint() * 1.5
        self.setFixedSize(new_size)


class TallPushButton(QPushButton):
    def __init__(self, text: str):
        super().__init__(text)
        new_height = int(self.sizeHint().height() * 1.5)
        self.setFixedHeight(new_height)


class BigCheckBox(QCheckBox):
    def __init__(self, text: str | None = None):
        super().__init__(text)
        settings = get_settings()
        font_size = settings.value(
            'font_size', defaultValue=DEFAULT_SETTINGS['font_size'], type=int)
        new_size = font_size * 1.5
        self.setStyleSheet(f'''
            QCheckBox::indicator {{ width: {new_size}px; height: {new_size}px; }}
            QCheckBox:disabled {{ color: #808080; }}
            QCheckBox::indicator:disabled {{
                background-color: #e0e0e0;
                border: 1px solid #a0a0a0;
            }}
            QCheckBox::indicator:disabled:checked {{
                background-color: #a0a0a0;
            }}
        ''')
