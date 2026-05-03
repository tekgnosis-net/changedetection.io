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
