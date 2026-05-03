"""Integration test: restock LLM-fallback vision call path.

Four scenarios:
  1. llm_use_for_restock=False → LLM never invoked
  2. llm_use_for_restock=True + vision off → text-only LLM
  3. llm_use_for_restock=True + vision on with screenshot → multipart
  4. llm_use_for_restock=True + vision on but no screenshot → text fall-through
"""
from unittest.mock import patch
import os
import io
from PIL import Image


def _setup_watch_for_restock(ds, with_screenshot=True):
    uuid = ds.add_watch(url='http://example.com', extras={'processor': 'restock_diff'})
    watch = ds.data['watching'][uuid]
    watch['paused'] = True  # Prevent worker pool race (live_server starts the worker).
    watch['fetch_backend'] = 'html_playwright'

    if with_screenshot:
        os.makedirs(watch.data_dir, exist_ok=True)
        buf = io.BytesIO()
        Image.new('RGB', (640, 480)).save(buf, format='PNG')
        with open(os.path.join(watch.data_dir, 'last-screenshot.png'), 'wb') as f:
            f.write(buf.getvalue())

    ds.data['settings']['application']['llm'] = {
        'model': 'openai/qwen3-vl-32b', 'api_key': 'sk',
        'api_base': 'http://10.0.20.64:8011/v1',
        'provider_kind': 'openai_compatible',
    }
    ds.data['settings']['application']['llm_restock_use_fallback_extract'] = True
    return uuid, watch


def test_restock_use_for_restock_false_skips_llm(client, live_server):
    ds = client.application.config['DATASTORE']
    _, watch = _setup_watch_for_restock(ds)
    watch['llm_use_for_restock'] = False

    from changedetectionio.processors.restock_diff.plugins import llm_restock
    llm_restock.datastore = ds

    with patch('changedetectionio.llm.client.completion') as mock_complete:
        result = llm_restock.run_llm_restock_extraction(watch, 'page text')
    assert result is None
    mock_complete.assert_not_called()


def test_restock_text_only_when_vision_off(client, live_server):
    ds = client.application.config['DATASTORE']
    _, watch = _setup_watch_for_restock(ds)
    watch['llm_use_for_restock'] = True
    watch['llm_use_vision'] = False

    from changedetectionio.processors.restock_diff.plugins import llm_restock
    llm_restock.datastore = ds

    captured = {}
    def fake(model, messages, **kw):
        captured['messages'] = messages
        return ('{"price": 10.0, "currency": "USD", "availability": "instock"}',
                50, 25, 25)

    with patch('changedetectionio.llm.client.completion', side_effect=fake):
        result = llm_restock.run_llm_restock_extraction(watch, 'page with $10')

    user_msg = next(m for m in captured['messages'] if m['role'] == 'user')
    assert isinstance(user_msg['content'], str)  # text-only
    assert result is not None


def test_restock_vision_when_screenshot_present(client, live_server):
    ds = client.application.config['DATASTORE']
    _, watch = _setup_watch_for_restock(ds, with_screenshot=True)
    watch['llm_use_for_restock'] = True
    watch['llm_use_vision'] = True
    watch['llm_vision_verified'] = True

    from changedetectionio.processors.restock_diff.plugins import llm_restock
    llm_restock.datastore = ds

    captured = {}
    def fake(model, messages, **kw):
        captured['messages'] = messages
        return ('{"price": 10.0, "currency": "USD", "availability": "instock"}',
                50, 25, 25)

    with patch('changedetectionio.llm.client.completion', side_effect=fake):
        result = llm_restock.run_llm_restock_extraction(watch, 'page text')

    user_msg = next(m for m in captured['messages'] if m['role'] == 'user')
    assert isinstance(user_msg['content'], list)  # multipart
    assert any(p.get('type') == 'image_url' for p in user_msg['content'])


def test_restock_vision_falls_back_when_no_screenshot(client, live_server):
    ds = client.application.config['DATASTORE']
    _, watch = _setup_watch_for_restock(ds, with_screenshot=False)
    watch['llm_use_for_restock'] = True
    watch['llm_use_vision'] = True
    watch['llm_vision_verified'] = True

    from changedetectionio.processors.restock_diff.plugins import llm_restock
    llm_restock.datastore = ds

    captured = {}
    def fake(model, messages, **kw):
        captured['messages'] = messages
        return ('{"price": 10.0, "currency": "USD", "availability": "instock"}',
                50, 25, 25)

    with patch('changedetectionio.llm.client.completion', side_effect=fake):
        result = llm_restock.run_llm_restock_extraction(watch, 'page text')

    user_msg = next(m for m in captured['messages'] if m['role'] == 'user')
    assert isinstance(user_msg['content'], str)  # text-only fall-through
