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
    watch['paused'] = True  # Prevent the worker pool (started by live_server) from racing this test
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

    captured = {}
    def fake_completion(model, messages, **kw):
        captured['messages'] = messages
        return ('Price card moved from $89 to $67', 100, 50, 50)

    with patch('changedetectionio.llm.client.completion', side_effect=fake_completion):
        from changedetectionio.llm.evaluator import summarise_change
        result = summarise_change(watch, ds, diff='- old\n+ new', current_snapshot='new')

    assert 'messages' in captured
    user_msg = next(m for m in captured['messages'] if m['role'] == 'user')
    assert isinstance(user_msg['content'], list)
    assert any(p.get('type') == 'image_url' for p in user_msg['content'])
    assert 'Price card' in result
