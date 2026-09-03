from src.rag.generator import MockProvider


def test_mock_provider_combines_facts_and_evidence():
    prompt = """QUESTION
Explain why revenue changed and calculate the growth.

STRUCTURED FACTS / CALCULATIONS
Revenue grew 6.43% year over year.

EVIDENCE
[1] Section: Item 7. Management's Discussion and Analysis
Net sales increased primarily due to higher Services net sales and product mix changes.

[2] Section: Item 1A. Risk Factors
The company faces competitive and macroeconomic risks.

Answer using only the information above."""

    result = MockProvider().generate("", prompt)

    assert "6.43%" in result.text
    assert "[1]" in result.text
    assert "Net sales increased" in result.text


def test_mock_provider_without_evidence_or_facts_is_safe():
    prompt = """QUESTION
What happened?

STRUCTURED FACTS / CALCULATIONS
None

EVIDENCE
None

Answer using only the information above."""

    result = MockProvider().generate("", prompt)

    assert (
        result.text
        == "Insufficient evidence in the retrieved SEC filing "
        "to answer this confidently."
    )
