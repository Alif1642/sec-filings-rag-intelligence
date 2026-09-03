from ingestion.filing_parser import FilingParser


def test_parser_extracts_sections_paragraphs_and_tables():
    html = """<html><body>
    <h2>Item 1A. Risk Factors</h2>
    <p>Market conditions may adversely affect the business.</p>
    <table><tr><th>Year</th><th>Revenue</th></tr><tr><td>2025</td><td>100</td></tr></table>
    </body></html>"""
    blocks = FilingParser().parse(html)
    assert any(b.section.lower().startswith('item 1a') for b in blocks)
    assert any('Market conditions' in b.text for b in blocks)
    assert any('Year | Revenue' in b.text for b in blocks)
