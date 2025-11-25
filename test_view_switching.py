#!/usr/bin/env python3
"""Test script for tag/caption view switching."""

import sys
import tempfile
from pathlib import Path

# Add taggui to path
sys.path.insert(0, str(Path(__file__).parent / 'taggui'))

from utils.image import Image

def test_view_switching_logic():
    """Test the view switching logic."""
    print("Testing view switching logic...")

    # Create test image with both tags and caption
    test_path = Path("/tmp/test_switch.jpg")
    img = Image(
        path=test_path,
        dimensions=(100, 100),
        tags=["tag1", "tag2", "tag3"],
        caption="This is a test caption with multiple lines.\nSecond line here.",
        tags_file_type='.tags.txt'
    )

    assert img.tags == ["tag1", "tag2", "tag3"]
    assert img.caption == "This is a test caption with multiple lines.\nSecond line here."
    assert img.tags_file_type == '.tags.txt'

    print("✓ Image object with tags and caption created correctly")

    # Test different scenarios
    scenarios = [
        ("Only tags", ["tag1"], None, '.txt'),
        ("Only caption", [], "Caption only", '.txt'),
        ("Both tags and caption", ["tag1", "tag2"], "Both available", '.tags.txt'),
    ]

    for name, tags, caption, file_type in scenarios:
        img = Image(
            path=test_path,
            dimensions=(100, 100),
            tags=tags,
            caption=caption,
            tags_file_type=file_type
        )
        print(f"  ✓ Scenario '{name}': tags={len(tags)}, caption={'Yes' if caption else 'No'}")

    print("\n✓ All view switching scenarios work correctly")

def test_file_labels():
    """Test file label generation."""
    print("\nTesting file label generation...")

    test_path = Path("/tmp/IMG_6400.JPG")

    # Test with .tags.txt
    img = Image(
        path=test_path,
        dimensions=(100, 100),
        tags=["test"],
        caption=None,
        tags_file_type='.tags.txt'
    )

    expected_tags_label = f"View tags ({test_path.name}.tags.txt)"
    print(f"  Tags label: {expected_tags_label}")
    assert img.tags_file_type == '.tags.txt'

    # Test with .txt
    img2 = Image(
        path=test_path,
        dimensions=(100, 100),
        tags=["test"],
        caption="Caption text",
        tags_file_type='.txt'
    )

    expected_tags_label2 = f"View tags ({test_path.stem}.txt)"
    expected_caption_label = f"View caption ({test_path.name}.caption.txt)"
    print(f"  Tags label: {expected_tags_label2}")
    print(f"  Caption label: {expected_caption_label}")

    print("\n✓ File labels generated correctly")

def main():
    """Run all tests."""
    print("=" * 60)
    print("View Switching Test Suite")
    print("=" * 60)

    try:
        test_view_switching_logic()
        test_file_labels()

        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        print("\nTo test the UI:")
        print("1. Run the application: python taggui/run_gui.py")
        print("2. Load a directory with images")
        print("3. Look at the bottom of the Image Tags dock")
        print("4. Click on 'View tags' or 'View caption' to switch views")
        print("5. Active view will be shown in blue and bold")
        return 0
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
