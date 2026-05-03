"""Load-bearing regression test: vision-off watches produce text-only LLM
messages, byte-identical to pre-vision behaviour.

Any future refactor that accidentally pulls vision-prep code into the
text-only path breaks this — for example, if someone moves
load_and_prepare_screenshot out of the `if use_vision:` guard, both
tests will fail because the user message would become multipart.

The assertion `isinstance(user_msg['content'], str)` is the contract:
no multipart leakage when vision is off.
"""
from unittest.mock import patch


def test_summarise_change_vision_off_messages_are_text_only(client, live_server):
    """text_json_diff: vision off → summarise_change builds text-only messages."""
    ds = client.application.config['DATASTORE']
    uuid = ds.add_watch(url='http://example.com')
    watch = ds.data['watching'][uuid]
    watch['paused'] = True  # Prevent worker pool race (live_server starts the worker).
    watch['llm_use_vision'] = False
    watch['fetch_backend'] = 'html_playwright'

    ds.data['settings']['application']['llm'] = {
        'model': 'openai/qwen3-vl-32b', 'api_key': 'sk', 'api_base': 'http://x',
        'provider_kind': 'openai_compatible',
    }

    captured = {}
    def fake(model, messages, **kw):
        captured['messages'] = messages
        return ('summary', 10, 5, 5)

    with patch('changedetectionio.llm.client.completion', side_effect=fake):
        from changedetectionio.llm.evaluator import summarise_change
        summarise_change(watch, ds, diff='diff', current_snapshot='snap')

    user_msg = next(m for m in captured['messages'] if m['role'] == 'user')
    assert isinstance(user_msg['content'], str), \
        "vision-off watches must produce text-only messages"


def test_restock_vision_off_messages_are_text_only(client, live_server):
    """restock_diff: vision off → run_llm_restock_extraction builds text-only messages."""
    ds = client.application.config['DATASTORE']
    uuid = ds.add_watch(url='http://example.com', extras={'processor': 'restock_diff'})
    watch = ds.data['watching'][uuid]
    watch['paused'] = True  # Prevent worker pool race.
    watch['llm_use_for_restock'] = True
    watch['llm_use_vision'] = False
    watch['fetch_backend'] = 'html_playwright'

    ds.data['settings']['application']['llm'] = {
        'model': 'openai/qwen3-vl-32b', 'api_key': 'sk', 'api_base': 'http://x',
        'provider_kind': 'openai_compatible',
    }
    ds.data['settings']['application']['llm_restock_use_fallback_extract'] = True

    # Inject datastore into the plugin module (same pattern as test_llm_restock_plugin.py).
    from changedetectionio.processors.restock_diff.plugins import llm_restock
    llm_restock.datastore = ds

    captured = {}
    def fake(model, messages, **kw):
        captured['messages'] = messages
        return ('{"price": 10}', 10, 5, 5)

    with patch('changedetectionio.llm.client.completion', side_effect=fake):
        llm_restock.run_llm_restock_extraction(watch, 'page text')

    user_msg = next(m for m in captured['messages'] if m['role'] == 'user')
    assert isinstance(user_msg['content'], str), \
        "vision-off restock watches must produce text-only messages"
