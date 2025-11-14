# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TagGUI is a cross-platform desktop application for quickly adding and editing image tags and captions, aimed towards creators of image datasets for generative AI models. Built with PySide6 (Qt for Python) and PyTorch/Transformers for AI captioning features.

## Development Commands

### Running the Application
```bash
python taggui/run_gui.py
```

Python 3.12 is recommended, but Python 3.11 should also work.

### Development Mode
Set the environment variable to enable detailed logging:
```bash
export TAGGUI_ENVIRONMENT=development
python taggui/run_gui.py
```

### Building with PyInstaller
```bash
pyinstaller taggui.spec
```

The spec file bundles the application with required resources (tokeniser, icon) and creates an executable in `dist/taggui/`.

### Installing Dependencies

**Standard Installation (Windows):**
```bash
pip install -r requirements.txt
```

**Linux Installation:**
On Linux, flash-attn requires building from source, which needs PyTorch to be installed first. Use this workaround:
```bash
# Install PyTorch first
pip install torch==2.8.0

# Install remaining dependencies (skip flash-attn if build fails)
pip install -r requirements.txt
```

If flash-attn fails to build, the application will work fine without it. FlashAttention is optional and only provides performance improvements for Florence-2 and Phi-3-Vision models. All other captioning models work without it.

**Note:** Requirements include platform-specific PyTorch and FlashAttention wheels for CUDA support on Windows. Linux uses PyPI packages.

## Architecture

### Core Components

**Models (MVC Pattern)**
- `ImageListModel`: Central model managing the list of images and their tags. Maintains undo/redo stacks for tag operations. All images are loaded from a directory into memory with their associated `.txt` tag files.
- `ProxyImageListModel`: Filtering layer for ImageListModel supporting complex filter syntax (tag:, caption:, name:, path:, logical operators).
- `ImageTagListModel`: String list model for tags of the currently selected image(s).
- `TagCounterModel`: Tracks usage frequency of all tags across images for autocomplete suggestions.

**Widgets (UI Components)**
- `MainWindow`: Primary window orchestrating all dock widgets and models. Handles menu actions and keyboard shortcuts.
- `ImageList`: Left dock showing thumbnails of all images in the loaded directory.
- `ImageTagsEditor`: Right dock for viewing/editing tags on selected image(s), with autocomplete and token counter.
- `AllTagsEditor`: Right dock showing all unique tags sorted by frequency, with batch rename/delete operations.
- `AutoCaptioner`: Right dock for AI-powered caption generation using various VLM models.
- `ImageViewer`: Central widget displaying the currently selected image.

**Auto-Captioning System**
- `AutoCaptioningModel`: Base class for all captioning models. Handles model loading, prompt template variables ({tags}, {name}, {directory}), generation parameters, and constrained generation (bad_words, forced_words).
- Model implementations in `auto_captioning/models/`: Each file implements a specific VLM (Florence-2, LLaVA variants, Phi-3-Vision, Moondream, JoyCaption, WD Tagger).
- `CaptioningThread`: QThread running caption generation in background to avoid blocking the UI.

### Data Flow

1. User loads directory → `ImageListModel` scans for images and `.txt` files → Creates `Image` objects with tags
2. Tag changes → `ImageListModel` updates image tags → Saves to `.txt` file → Notifies views → Updates `TagCounterModel`
3. Filtering → User enters filter expression → `ProxyImageListModel` parses and applies filter → Updates visible images
4. Auto-captioning → User configures model and parameters → `CaptioningThread` loads model → Generates captions → Updates image tags

### Key Patterns

**Settings Management**
- Uses `QSettings` for persistent configuration stored in platform-specific locations.
- `DEFAULT_SETTINGS` dict in `utils/settings.py` defines fallback values.
- Common settings: tag_separator, font_size, image_list_image_width, autocomplete_tags, models_directory_path.

**Undo/Redo**
- `ImageListModel` maintains deque-based undo/redo stacks with configurable size (32 items).
- Each history item stores action name, full tag state, and whether confirmation is needed.
- All tag modifications (add, delete, rename, reorder, batch operations) go through this system.

**Image Loading**
- Recursive directory scanning with support for subdirectories.
- Lazy thumbnail generation cached in `Image` objects.
- Tags loaded from `.txt` files with same base name as images.
- Supports common formats: bmp, gif, jpg, jpeg, png, tif, tiff, webp.

**PyInstaller Bundling**
- Resources accessed via `get_resource_path()` which handles both development and bundled modes.
- Bundled resources: CLIP tokeniser (`clip-vit-base-patch32`), application icon.
- Hidden imports configured in `taggui.spec` for dynamic model dependencies (e.g., timm.models.layers).

## Important Implementation Details

- Tag separator is configurable (default `,`) with optional space insertion.
- Token counting uses CLIP tokeniser for Stable Diffusion compatibility (75 token limit).
- Auto-captioning supports GPU (CUDA) and CPU, with optional 4-bit quantisation for memory efficiency.
- Filter syntax supports wildcards (`*`, `?`), logical operators (AND, OR, NOT), and numeric comparisons for tag/char/token counts.
- Keyboard shortcuts are extensively used for efficiency (see README Controls section).
- All tag operations on multiple images require confirmation dialogs.
- The application clears QSettings and shows error dialog on uncaught exceptions.
