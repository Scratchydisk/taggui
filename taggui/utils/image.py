from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtGui import QIcon


@dataclass
class Image:
    path: Path
    dimensions: tuple[int, int] | None
    tags: list[str] = field(default_factory=list)
    caption: str | None = None
    tags_file_type: str = '.txt'  # Track which file type was loaded: '.txt', '.tags.txt', or '.caption.txt'
    thumbnail: QIcon | None = None
