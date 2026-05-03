"""Integration test: text_json_diff change-summary vision call path."""
from unittest.mock import patch
import os
import io
from PIL import Image


def test_summary_uses_vision_when_flags_on(client, live_server, datastore_path):
    """Watch with vision flags on + browser fetcher + screenshot present
    → summarise_change builds multipart messages with image_url part."""
    ds = client.application.config['DATASTORE']
    uuid = ds.add_watch(url='http://example.com')
    watch = ds.data['watching'][uuid]
    watch['llm_use_vision'] = True
    watch['llm_vision_verified'] = True
    watch['fetch_backend'] = 'html_playwright'

    os.makedirs(watch.data_dir, exist_ok=True)
    buf = io.BytesIO()
    Image.new('RGB', (640, 480), color=(200, 200, 200)).save(buf, format='PNG')
    screenshot_path = os.path.join(watch.data_dir, 'last-screenshot.png')
    with open(screenshot_path, 'wb') as f:
        f.write(buf.getvalue())

    ds.data['settings']['application']['llm'] = {
        'model': 'openai/qwen3-vl-32b', 'api_key': 'sk-test',
        'api_base': 'http://10.0.20.64:8011/v1',
        'provider_kind': 'openai_compatible',
    }

    # Only capture calls that contain vision (multipart) messages — the
    # background worker may concurrently call fake_completion with text-only
    # messages, so we ignore those to avoid a race overwriting our result.
    captured = {}
    def fake_completion(model, messages, **kw):
        has_vision = any(
            isinstance(m.get('content'), list) for m in messages
        )
        if has_vision:
            captured['messages'] = messages
        return ('Price card moved from $89 to $67', 100, 50, 50)

    with patch('changedetectionio.llm.client.completion', side_effect=fake_completion):
        from changedetectionio.llm.evaluator import summarise_change
        # Re-write the screenshot in case the background worker consumed it
        buf2 = io.BytesIO()
        Image.new('RGB', (640, 480), color=(200, 200, 200)).save(buf2, format='PNG')
        with open(screenshot_path, 'wb') as f:
            f.write(buf2.getvalue())
        result = summarise_change(watch, ds, diff='- old\n+ new', current_snapshot='new')

    assert 'messages' in captured
    user_msg = next(m for m in captured['messages'] if m['role'] == 'user')
    assert isinstance(user_msg['content'], list)
    assert any(p.get('type') == 'image_url' for p in user_msg['content'])
    assert 'Price card' in result
