"""Unit tests for changedetectionio.llm.vision."""
import base64
import io
import random
from unittest.mock import MagicMock, patch

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


def test_preprocess_with_hint_fast_path(monkeypatch):
    """Valid hint matching current context → ladder is NOT iterated.
    PIL.Image.save is called exactly once.
    Non-default hint values prove the fast-path was taken."""
    src = _make_png(1280, 720)
    context = {
        'model': 'openai/qwen3-vl-32b',
        'fetcher_backend': 'html_playwright',
        'api_base': 'http://10.0.20.64:8011/v1',
        'provider_kind': 'openai_compatible',
    }
    # Use non-default hint dimensions to force distinguishable post-condition
    hint = {**context, 'quality': 70, 'max_width': 800, 'max_height': 4096}

    save_call_count = [0]
    real_save = Image.Image.save

    def counting_save(self, *a, **kw):
        save_call_count[0] += 1
        return real_save(self, *a, **kw)

    monkeypatch.setattr(Image.Image, 'save', counting_save)

    out_bytes, mime, used_hint = vision.preprocess_screenshot(
        src, hint=hint, context=context
    )
    assert mime == 'image/jpeg'
    assert used_hint['model'] == context['model']
    # Fast-path: hinted; Ladder default: 85
    assert used_hint['quality'] == 70
    # Fast-path: hinted; Ladder default: 1280
    assert used_hint['max_width'] == 800
    assert save_call_count[0] == 1, "Ladder must not iterate when hint works"


def test_preprocess_with_stale_context_discards_hint():
    """Hint's embedded model differs from current context's model →
    hint is silently discarded; ladder runs from defaults; used_hint
    reflects the new (current) context."""
    src = _make_png(1280, 720)
    stale_hint = {
        'model': 'openai/qwen3-vl-OLD',
        'fetcher_backend': 'html_playwright',
        'api_base': 'http://10.0.20.64:8011/v1',
        'provider_kind': 'openai_compatible',
        'quality': 65,
        'max_width': 1024,
        'max_height': 3000,
    }
    current_context = {
        'model': 'openai/qwen3-vl-NEW',
        'fetcher_backend': 'html_playwright',
        'api_base': 'http://10.0.20.64:8011/v1',
        'provider_kind': 'openai_compatible',
    }
    out_bytes, mime, used_hint = vision.preprocess_screenshot(
        src, hint=stale_hint, context=current_context
    )
    assert used_hint['model'] == current_context['model']
    assert used_hint['quality'] == vision.VISION_JPEG_QUALITY


def test_preprocess_with_changed_api_base_discards_hint():
    """Different api_base in current context → hint discarded."""
    src = _make_png(1280, 720)
    hint = {
        'model': 'openai/qwen3-vl', 'fetcher_backend': 'html_playwright',
        'api_base': 'http://10.0.20.64:8011/v1',
        'provider_kind': 'openai_compatible',
        'quality': 65, 'max_width': 1024, 'max_height': 3000,
    }
    current_context = {
        'model': 'openai/qwen3-vl', 'fetcher_backend': 'html_playwright',
        'api_base': 'http://10.0.20.65:8012/v1',  # different port
        'provider_kind': 'openai_compatible',
    }
    _, _, used_hint = vision.preprocess_screenshot(
        src, hint=hint, context=current_context
    )
    assert used_hint['api_base'] == current_context['api_base']
    assert used_hint['quality'] == vision.VISION_JPEG_QUALITY  # ladder ran fresh


def test_load_and_prepare_returns_none_when_no_screenshot(tmp_path):
    """No screenshot on disk → returns None (caller falls back to text)."""
    watch = MagicMock()
    watch.data_dir = str(tmp_path)
    watch.get = MagicMock(return_value=None)
    llm_cfg = {
        'model': 'openai/qwen3-vl', 'api_base': 'http://x',
        'provider_kind': 'openai_compatible',
    }
    result = vision.load_and_prepare_screenshot(watch, llm_cfg)
    assert result is None


def test_load_and_prepare_returns_none_for_corrupt_image(tmp_path):
    """Non-image bytes (e.g. html_requests response body) → returns None."""
    (tmp_path / 'last-screenshot.png').write_bytes(b'<html>not an image</html>')
    watch = MagicMock()
    watch.data_dir = str(tmp_path)
    watch.get = MagicMock(return_value=None)
    llm_cfg = {'model': 'm', 'api_base': 'b', 'provider_kind': 'openai_compatible'}
    result = vision.load_and_prepare_screenshot(watch, llm_cfg)
    assert result is None


def test_load_and_prepare_persists_hint_on_success(tmp_path):
    """Successful preprocess updates watch['llm_vision_preprocess_hint']."""
    src = _make_png(1280, 720)
    (tmp_path / 'last-screenshot.png').write_bytes(src)

    class FakeWatch(dict):
        def __init__(self, data, dir):
            super().__init__(data)
            self.data_dir = dir

    watch = FakeWatch({
        'llm_vision_preprocess_hint': None,
        'fetch_backend': 'html_playwright',
    }, str(tmp_path))
    llm_cfg = {
        'model': 'openai/qwen3-vl-32b',
        'api_base': 'http://10.0.20.64:8011/v1',
        'provider_kind': 'openai_compatible',
    }
    result = vision.load_and_prepare_screenshot(watch, llm_cfg)
    assert result is not None
    bytes_out, mime = result
    assert mime == 'image/jpeg'
    assert watch['llm_vision_preprocess_hint'] is not None
    assert watch['llm_vision_preprocess_hint']['model'] == 'openai/qwen3-vl-32b'


def test_build_vision_messages_with_system_prompt():
    """Multipart messages: system + user-with-image."""
    msgs = vision.build_vision_messages(
        text_user_content="What's the price?",
        image_bytes=b'\xff\xd8\xff\xe0FAKE',
        mime_type='image/jpeg',
        system_prompt='You are a price extractor.',
    )
    assert len(msgs) == 2
    assert msgs[0] == {'role': 'system', 'content': 'You are a price extractor.'}
    assert msgs[1]['role'] == 'user'
    assert isinstance(msgs[1]['content'], list)
    assert len(msgs[1]['content']) == 2
    assert msgs[1]['content'][0] == {'type': 'text', 'text': "What's the price?"}
    assert msgs[1]['content'][1]['type'] == 'image_url'
    assert msgs[1]['content'][1]['image_url']['url'].startswith(
        'data:image/jpeg;base64,'
    )


def test_build_vision_messages_no_system_prompt():
    msgs = vision.build_vision_messages(
        text_user_content="hi", image_bytes=b'\xff\xd8FAKE',
    )
    assert len(msgs) == 1
    assert msgs[0]['role'] == 'user'


def test_build_vision_messages_previous_screenshot_unused_in_phase1():
    """previous_screenshot kwarg reserved; currently ignored."""
    msgs = vision.build_vision_messages(
        text_user_content="Compare", image_bytes=b'CURR',
        previous_screenshot=(b'PREV', 'image/jpeg'),
    )
    image_parts = [p for p in msgs[0]['content'] if p['type'] == 'image_url']
    assert len(image_parts) == 1


def test_probe_vision_capability_success():
    """litellm returns text → (True, message)."""
    fake_response = MagicMock()
    fake_response.choices = [MagicMock()]
    fake_response.choices[0].message.content = "I see a small grey square."
    fake_response.choices[0].finish_reason = 'stop'

    with patch('litellm.completion', return_value=fake_response):
        ok, msg = vision.probe_vision_capability(
            model='openai/qwen3-vl-32b', api_key='sk',
            api_base='http://10.0.20.64:8011/v1',
        )
    assert ok is True
    assert 'square' in msg.lower() or 'grey' in msg.lower()


def test_probe_vision_capability_endpoint_rejects():
    """litellm raises → (False, error message)."""
    err = Exception("Model does not support image inputs")
    with patch('litellm.completion', side_effect=err):
        ok, msg = vision.probe_vision_capability(
            model='openai/qwen3-32b-instruct', api_key='sk',
            api_base='http://10.0.20.64:8011/v1',
        )
    assert ok is False
    assert 'image' in msg.lower() or 'support' in msg.lower()


def test_probe_vision_capability_empty_content():
    """200 response but empty content → (False, helpful message)."""
    fake_response = MagicMock()
    fake_response.choices = [MagicMock()]
    fake_response.choices[0].message.content = ''
    fake_response.choices[0].finish_reason = 'length'

    with patch('litellm.completion', return_value=fake_response):
        ok, msg = vision.probe_vision_capability(
            model='openai/qwen3-vl-32b', api_key='sk',
            api_base='http://10.0.20.64:8011/v1',
        )
    assert ok is False
    assert 'empty' in msg.lower() or 'finish_reason' in msg.lower()
