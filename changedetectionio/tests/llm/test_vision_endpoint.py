"""Integration tests for the /settings/llm/vision-test route."""
from unittest.mock import patch
from flask import url_for


def test_vision_test_route_success(client, live_server):
    """Probe succeeds → returns {ok: true, text}."""
    ds = client.application.config['DATASTORE']
    ds.data['settings']['application']['llm'] = {
        'model': 'openai/qwen3-vl-32b', 'api_key': 'sk-test',
        'api_base': 'http://10.0.20.64:8011/v1',
        'provider_kind': 'openai_compatible',
    }
    with patch('changedetectionio.llm.vision.probe_vision_capability',
               return_value=(True, "I see a small grey square.")):
        resp = client.get(url_for('settings.llm.llm_vision_test'))
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload['ok'] is True


def test_vision_test_route_failure(client, live_server):
    """Probe failure → 400 with {ok: false, error}."""
    ds = client.application.config['DATASTORE']
    ds.data['settings']['application']['llm'] = {
        'model': 'openai/qwen3-32b-instruct', 'api_key': 'sk',
        'api_base': 'http://10.0.20.64:8011/v1',
        'provider_kind': 'openai_compatible',
    }
    with patch('changedetectionio.llm.vision.probe_vision_capability',
               return_value=(False, "Model does not support images")):
        resp = client.get(url_for('settings.llm.llm_vision_test'))
    assert resp.status_code == 400
    assert resp.get_json()['ok'] is False


def test_vision_test_route_no_model(client, live_server):
    """No LLM model configured → 400 with helpful error."""
    ds = client.application.config['DATASTORE']
    ds.data['settings']['application'].pop('llm', None)
    resp = client.get(url_for('settings.llm.llm_vision_test'))
    assert resp.status_code == 400
    assert 'No model configured' in resp.get_json()['error']


def test_per_watch_form_persists_vision_fields(client, live_server):
    """Saving a watch with vision toggle on stores it in the watch dict."""
    ds = client.application.config['DATASTORE']
    uuid = ds.add_watch(url='https://example.com')
    resp = client.post(
        url_for('ui.ui_edit.edit_page', uuid=uuid),
        data={
            'url': 'https://example.com',
            'tags': '',
            'time_between_check-hours': '1',
            'time_between_check-minutes': '0',
            'time_between_check-seconds': '0',
            'time_between_check-weeks': '0',
            'time_between_check-days': '0',
            'fetch_backend': 'system',
            'processor': 'text_json_diff',
            'method': 'GET',
            'extract_title_as_title': 'y',
            'llm_use_vision': 'y',
            'llm_vision_verified': '1',
            'llm_use_for_restock': 'true',
        },
        follow_redirects=True,
    )
    assert resp.status_code in (200, 302)
    watch = ds.data['watching'][uuid]
    assert watch.get('llm_use_vision') is True
    assert watch.get('llm_use_for_restock') is True
