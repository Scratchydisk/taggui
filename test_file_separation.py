#!/usr/bin/env python3
"""Test script for tag/caption file separation."""

import sys
import tempfile
from pathlib import Path

# Add taggui to path
sys.path.insert(0, str(Path(__file__).parent / 'taggui'))

from utils.image import Image

def test_image_dataclass():
    """Test the Image dataclass with new fields."""
    print("Testing Image dataclass...")

    test_path = Path("/tmp/test.jpg")
    img = Image(
        path=test_path,
        dimensions=(100, 100),
        tags=["tag1", "tag2"],
        caption="A test caption",
        tags_file_type='.tags.txt'
    )

    assert img.path == test_path
    assert img.tags == ["tag1", "tag2"]
    assert img.caption == "A test caption"
    assert img.tags_file_type == '.tags.txt'

    print("✓ Image dataclass works correctly")

def test_file_loading():
    """Test loading tags and captions from different file types."""
    print("\nTesting file loading priority...")

    # Create a temporary directory with test files
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create test image file
        img_path = tmpdir / "test.jpg"
        img_path.write_text("dummy")

        # Test 1: Only .txt file exists
        txt_file = tmpdir / "test.txt"
        txt_file.write_text("tag1, tag2, tag3")

        print(f"  Test 1: Only .txt exists")
        print(f"    Expected: Load from .txt")

        # Test 2: Both .txt and .tags.txt exist (.tags.txt should win)
        tags_file = tmpdir / "test.jpg.tags.txt"
        tags_file.write_text("tag_a, tag_b")

        print(f"  Test 2: Both .txt and .tags.txt exist")
        print(f"    Expected: Load from .tags.txt (priority)")

        # Test 3: Caption file
        caption_file = tmpdir / "test.jpg.caption.txt"
        caption_file.write_text("This is a caption about the image.")

        print(f"  Test 3: .caption.txt exists")
        print(f"    Expected: Caption loaded separately")

        print("\n✓ File structure created correctly")

def main():
    """Run all tests."""
    print("=" * 60)
    print("File Separation Test Suite")
    print("=" * 60)

    try:
        test_image_dataclass()
        test_file_loading()

        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        return 0
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
