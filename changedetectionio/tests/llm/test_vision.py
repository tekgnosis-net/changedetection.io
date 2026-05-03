"""Unit tests for changedetectionio.llm.vision."""
import base64
import io  # noqa: F401

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
