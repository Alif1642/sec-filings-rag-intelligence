from ingestion.company_resolver import CompanyResolver


class StubClient:
    def resolve_ticker(self, ticker):
        return {'ticker': ticker.upper(), 'cik': '0000000001', 'title': 'Example'}


def test_company_resolver_delegates_to_sec_client():
    assert CompanyResolver(StubClient()).resolve('abc')['ticker'] == 'ABC'
