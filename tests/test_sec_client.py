import json

from ingestion.sec_client import SECClient
from src.config import Settings


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self.status_code = status_code
        self.content = json.dumps(payload).encode()

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


class FakeSession:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def get(self, url, timeout):
        self.calls.append((url, timeout))
        return FakeResponse(self.payload)


def test_resolve_ticker_uses_official_mapping(tmp_path):
    payload = {'0': {'cik_str': 320193, 'ticker': 'AAPL', 'title': 'Apple Inc.'}}
    settings = Settings(data_dir=tmp_path, sec_user_agent='Tester test@example.com')
    client = SECClient(settings, session=FakeSession(payload))
    company = client.resolve_ticker('aapl')
    assert company == {'ticker': 'AAPL', 'cik': '0000320193', 'title': 'Apple Inc.'}


def test_normalize_cik():
    assert SECClient.normalize_cik('320193') == '0000320193'
