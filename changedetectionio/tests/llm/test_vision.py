"""Unit tests for changedetectionio.llm.vision."""
import base64
import io
import random
from unittest.mock import MagicMock

import pytest
from PIL import Image

from changedetectionio.llm import vision


def test_encode_as_data_url_jpeg():
    """encode_as_data_url wraps bytes in OpenAI-format data URL."""
    raw = b'\xff\xd8\xff\xe0FAKEJPEG'
    url = vision.encode_as_data_url(raw, 'image/jpeg')
    assert url.startswith('data:image/jpeg;base64,')
    decoded = base64.b64decode(url.split(',', 1)[1])
    assert decoded == raw


def test_encode_as_data_url_png():
    """Mime type is preserved in the data URL."""
    raw = b'\x89PNG\r\n\x1a\nFAKEPNG'
    url = vision.encode_as_data_url(raw, 'image/png')
    assert url.startswith('data:image/png;base64,')


def test_load_screenshot_missing_returns_none(tmp_path):
    """No last-screenshot.png on disk → returns None, INFO logged."""
    watch = MagicMock()
    watch.data_dir = str(tmp_path)  # empty dir
    result = vision.load_screenshot(watch)
    assert result is None


def test_load_screenshot_present_returns_bytes(tmp_path):
    """last-screenshot.png present → returns its bytes verbatim."""
    payload = b'\x89PNG\r\n\x1a\nNOT-DECODED-BY-LOAD-FUNCTION'
    (tmp_path / 'last-screenshot.png').write_bytes(payload)
    watch = MagicMock()
    watch.data_dir = str(tmp_path)
    result = vision.load_screenshot(watch)
    assert result == payload


def _make_png(width, height, color=(128, 128, 128)) -> bytes:
    """Synthesize a solid-color PNG of given dimensions for tests."""
    buf = io.BytesIO()
    Image.new('RGB', (width, height), color=color).save(buf, format='PNG')
    return buf.getvalue()


def test_preprocess_normal_screenshot_returns_jpeg_under_cap():
    """1280x720 PNG → JPEG output, dimensions preserved, under max_kb."""
    src = _make_png(1280, 720)
    out_bytes, mime, _hint = vision.preprocess_screenshot(src)
    assert mime == 'image/jpeg'
    img = Image.open(io.BytesIO(out_bytes))
    assert img.size == (1280, 720)
    assert len(out_bytes) <= vision.VISION_IMAGE_MAX_KB * 1024


def test_preprocess_full_page_top_cropped():
    """1280x20000 (full-page) PNG → 1280x4096 JPEG (top-cropped).
    Below-the-fold content is dropped; this is documented behavior."""
    src = _make_png(1280, 20000)
    out_bytes, mime, _hint = vision.preprocess_screenshot(src)
    img = Image.open(io.BytesIO(out_bytes))
    assert img.size == (1280, vision.VISION_IMAGE_MAX_HEIGHT)


def test_preprocess_oversize_width_resized_proportionally():
    """3840x2160 (4K) → 1280xN (proportional resize to width cap)."""
    src = _make_png(3840, 2160)
    out_bytes, _, _ = vision.preprocess_screenshot(src)
    img = Image.open(io.BytesIO(out_bytes))
    assert img.width == vision.VISION_IMAGE_MAX_WIDTH
    expected_height = int(2160 * (vision.VISION_IMAGE_MAX_WIDTH / 3840))
    assert abs(img.height - expected_height) <= 1


def test_preprocess_rgba_handled():
    """RGBA PNG → flattens to RGB JPEG."""
    buf = io.BytesIO()
    Image.new('RGBA', (1000, 800), color=(255, 0, 0, 128)).save(buf, format='PNG')
    out_bytes, mime, _ = vision.preprocess_screenshot(buf.getvalue())
    assert mime == 'image/jpeg'
    img = Image.open(io.BytesIO(out_bytes))
    assert img.mode == 'RGB'


def test_preprocess_returns_used_hint_dict():
    """Third return value is the hint dict; caller persists it on the watch."""
    src = _make_png(1280, 720)
    _, _, used_hint = vision.preprocess_screenshot(src)
    assert isinstance(used_hint, dict)
    assert 'quality' in used_hint
    assert 'max_width' in used_hint
    assert 'max_height' in used_hint


def test_preprocess_oversize_quality_ladder_kicks_in():
    """High-entropy image with tight cap → ladder descends past q=85."""
    random.seed(42)
    pixels = bytes(random.randint(0, 255) for _ in range(1280 * 4096 * 3))
    img = Image.frombytes('RGB', (1280, 4096), pixels)
    buf = io.BytesIO()
    img.save(buf, format='PNG', compress_level=0)
    src = buf.getvalue()

    out_bytes, _, used_hint = vision.preprocess_screenshot(src, max_kb=600)
    assert len(out_bytes) <= 600 * 1024
    # The ladder ran — quality reduced below the default 85
    assert used_hint['quality'] <= 85


def test_preprocess_unrescuable_raises():
    """Cap so tight no quality + dim combination satisfies → raises."""
    random.seed(7)
    pixels = bytes(random.randint(0, 255) for _ in range(1280 * 4096 * 3))
    img = Image.frombytes('RGB', (1280, 4096), pixels)
    buf = io.BytesIO()
    img.save(buf, format='PNG', compress_level=0)
    src = buf.getvalue()

    with pytest.raises(vision.VisionImageTooLargeError):
        vision.preprocess_screenshot(src, max_kb=5, max_width=1280)
