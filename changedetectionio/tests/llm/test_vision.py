"""Unit tests for changedetectionio.llm.vision."""
import base64
import io  # noqa: F401
from unittest.mock import MagicMock

from PIL import Image  # noqa: F401

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
