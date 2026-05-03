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


def test_restock_extras_extracted_and_persisted(client, live_server):
    """User-defined extras are extracted from the LLM JSON and stored on the watch."""
    ds = client.application.config['DATASTORE']
    _, watch = _setup_watch_for_restock(ds)
    watch['llm_use_for_restock'] = True
    watch['llm_use_vision'] = False
    watch['llm_extract_extras'] = 'Detect SALE banner and original price if struck through.'

    from changedetectionio.processors.restock_diff.plugins import llm_restock
    llm_restock.datastore = ds

    captured = {}
    def fake(model, messages, **kw):
        captured['messages'] = messages
        return ('{"price": 129.0, "currency": "AUD", "availability": "instock", '
                '"sale_active": true, "original_price": 289.99, "sale_label": "SALE"}',
                50, 25, 25)

    with patch('changedetectionio.llm.client.completion', side_effect=fake):
        result = llm_restock.run_llm_restock_extraction(watch, 'page text')

    # Core keys remain in the returned dict
    assert result.get('price') == 129.0
    assert result.get('availability') == 'instock'
    # Extras separated and persisted to the watch
    assert watch.get('llm_extracted_extras') == {
        'sale_active': True,
        'original_price': 289.99,
        'sale_label': 'SALE',
    }
    # The system prompt should include the user's directive
    sys_msg = captured['messages'][0]['content']
    assert 'Detect SALE banner' in sys_msg


def test_restock_vision_cues_only_when_vision_used(client, live_server):
    """VISION_CUES_PROMPT is included only when the request actually goes
    through the vision branch (multipart messages)."""
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
        return ('{"price": 129.0, "currency": "AUD", "availability": "instock"}',
                50, 25, 25)

    with patch('changedetectionio.llm.client.completion', side_effect=fake):
        llm_restock.run_llm_restock_extraction(watch, 'page text')

    sys_msg = captured['messages'][0]['content']
    assert 'VISUAL CUES' in sys_msg, "vision branch should include VISION_CUES_PROMPT"
    assert 'strikethrough' in sys_msg.lower(), "vision cues should mention strikethrough handling"


def test_restock_no_vision_cues_in_text_only(client, live_server):
    """Counterpart: text-only watches do NOT get VISION_CUES_PROMPT."""
    ds = client.application.config['DATASTORE']
    _, watch = _setup_watch_for_restock(ds)
    watch['llm_use_for_restock'] = True
    watch['llm_use_vision'] = False

    from changedetectionio.processors.restock_diff.plugins import llm_restock
    llm_restock.datastore = ds

    captured = {}
    def fake(model, messages, **kw):
        captured['messages'] = messages
        return ('{"price": 129.0, "currency": "AUD", "availability": "instock"}',
                50, 25, 25)

    with patch('changedetectionio.llm.client.completion', side_effect=fake):
        llm_restock.run_llm_restock_extraction(watch, 'page text')

    sys_msg = captured['messages'][0]['content']
    assert 'VISUAL CUES' not in sys_msg, "text-only branch must NOT include VISION_CUES_PROMPT"


def test_restock_llm_use_for_restock_true_invokes_llm_with_extras_and_vision_cues(client, live_server):
    """Integration shape test: when llm_use_for_restock=True + vision on +
    extras filled, the LLM call sends multipart messages with the vision
    cues paragraph AND the extras directive in the system prompt — the
    full configured stack as the user would set it up."""
    ds = client.application.config['DATASTORE']
    _, watch = _setup_watch_for_restock(ds, with_screenshot=True)
    watch['llm_use_for_restock'] = True
    watch['llm_use_vision'] = True
    watch['llm_vision_verified'] = True
    watch['llm_extract_extras'] = 'Detect SALE banner and original price if struck through.'

    captured = {}
    def fake(model, messages, **kw):
        captured['messages'] = messages
        return ('{"price": 129.0, "currency": "AUD", "availability": "instock", '
                '"sale_active": true, "original_price": 289.99, "sale_label": "SALE"}',
                100, 50, 50)

    with patch('changedetectionio.llm.client.completion', side_effect=fake):
        from changedetectionio.processors.restock_diff.plugins import llm_restock
        result = llm_restock.run_llm_restock_extraction(watch, 'page text with multiple prices')

    # Multipart user message (vision branch took)
    user_msg = next(m for m in captured['messages'] if m['role'] == 'user')
    assert isinstance(user_msg['content'], list)
    assert any(p.get('type') == 'image_url' for p in user_msg['content'])

    # System prompt has both VISION_CUES and the user's extras directive
    sys_msg = captured['messages'][0]['content']
    assert 'VISUAL CUES' in sys_msg
    assert 'Detect SALE banner' in sys_msg

    # Result has the right price and the extras persisted
    assert result.get('price') == 129.0
    assert watch.get('llm_extracted_extras') == {
        'sale_active': True,
        'original_price': 289.99,
        'sale_label': 'SALE',
    }


def test_restock_max_tokens_sized_for_reasoning_models(client, live_server):
    """Regression: reasoning models burn output budget on chain-of-thought.
    The base must be large enough that text_len=0 / finish_reason='length'
    isn't the failure mode for normal sale-page extractions."""
    ds = client.application.config['DATASTORE']
    _, watch = _setup_watch_for_restock(ds)
    watch['llm_use_for_restock'] = True
    watch['llm_use_vision'] = False

    captured_kwargs = {}
    def fake(model, messages, **kw):
        captured_kwargs.update(kw)
        return ('{"price": 129.0, "currency": "AUD", "availability": "instock"}',
                50, 25, 25)

    with patch('changedetectionio.llm.client.completion', side_effect=fake):
        from changedetectionio.processors.restock_diff.plugins import llm_restock
        llm_restock.run_llm_restock_extraction(watch, 'page text')

    # Without provider_kind=openai_compatible the multiplier is 1x → max_tokens=800.
    # With provider_kind=openai_compatible the multiplier is 5x → max_tokens=4000.
    # The configured llm has provider_kind=openai_compatible (set in the helper),
    # so this should be the multiplied value.
    assert captured_kwargs.get('max_tokens') == 4000, \
        f"Expected max_tokens=4000 (800 * 5x for openai_compatible), got {captured_kwargs.get('max_tokens')}"
