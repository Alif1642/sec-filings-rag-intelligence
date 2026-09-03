from src.rag.answer_schema import Citation
from src.rag.citation import citation_completeness, validate_citations


def test_citation_validation_and_completeness():
    citations = [Citation(citation_id='1', chunk_id='x', source_url='https://www.sec.gov/x', section='Risk Factors')]
    ok, invalid = validate_citations('Revenue was $10 million. [1]', citations)
    assert ok and invalid == []
    assert citation_completeness('Revenue was $10 million. [1]') == 1.0


def test_invalid_citation_detected():
    citations = [Citation(citation_id='1', chunk_id='x', source_url='https://www.sec.gov/x', section='Risk Factors')]
    ok, invalid = validate_citations('Claim. [2]', citations)
    assert not ok and invalid == ['2']
