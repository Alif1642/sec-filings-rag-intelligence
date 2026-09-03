from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

import httpx

from src.config import Settings, get_settings
from src.rag.prompts import SYSTEM_PROMPT, build_user_prompt


@dataclass(slots=True)
class GenerationResult:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ''


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> GenerationResult:
        raise NotImplementedError


class MockProvider(LLMProvider):
    """Non-fabricating demo provider that extracts relevant SEC evidence."""

    def generate(self, system_prompt: str, user_prompt: str) -> GenerationResult:
        question_match = re.search(
            r'QUESTION\n(.+?)\n\nSTRUCTURED FACTS',
            user_prompt,
            flags=re.S,
        )
        question = question_match.group(1).strip() if question_match else ''

        evidence_match = re.search(
            r'EVIDENCE\n(.+?)\n\nAnswer using',
            user_prompt,
            flags=re.S,
        )
        evidence = evidence_match.group(1).strip() if evidence_match else ''

        fact_match = re.search(
            r'STRUCTURED FACTS / CALCULATIONS\n(.+?)\n\nEVIDENCE',
            user_prompt,
            flags=re.S,
        )
        facts = fact_match.group(1).strip() if fact_match else ''

        evidence_summaries: list[tuple[int, str]] = []

        if evidence and evidence != 'None':
            blocks = re.findall(
                r'(\[\d+\].*?)(?=\n\n\[\d+\]|\Z)',
                evidence,
                flags=re.S,
            )

            question_lower = question.lower()

            for block in blocks:
                cid_match = re.match(r'\[(\d+)\]', block)
                section_match = re.match(
                    r'\[\d+\]\s*Section:\s*([^\n]+)',
                    block,
                )

                if not cid_match:
                    continue

                citation_id = cid_match.group(1)
                section = (
                    section_match.group(1).strip()
                    if section_match
                    else ''
                )

                body = re.sub(
                    r'^\[\d+\]\s*Section:[^\n]*\n',
                    '',
                    block,
                )

                # For revenue/sales questions, directly extract causal
                # "net sales increased/decreased ... due to ..." statements.
                # This avoids treating tables + narrative as one huge sentence.
                if (
                    'revenue' in question_lower
                    or 'sales' in question_lower
                    or 'growth' in question_lower
                ):
                    direct_sales_matches = re.findall(
                        r'([^.\n]{0,80}\bnet sales\s+'
                        r'(?:increased|decreased)\s+during\s+\d{4}\s+'
                        r'compared to\s+\d{4}\s+(?:primarily\s+)?due to\s+'
                        r'[^.\n]{10,350}\.)',
                        body,
                        flags=re.I,
                    )

                    for direct_sentence in direct_sales_matches:
                        cleaned_direct = re.sub(
                            r'\s+',
                            ' ',
                            direct_sentence,
                        ).strip()

                        evidence_summaries.append(
                            (
                                100,
                                f"{cleaned_direct[:500]} [{citation_id}]",
                            )
                        )

                sentences = [
                    sentence.strip()
                    for sentence in re.split(
                        r'(?<=[.!?])\s+',
                        body.replace('\n', ' '),
                    )
                    if len(sentence.strip()) > 60
                ]

                for sentence in sentences:
                    lowered = sentence.lower()
                    score = 0

                    # Strongly prefer direct explanations of sales/revenue changes.
                    if 'net sales increased' in lowered:
                        score += 20
                    if 'net sales decreased' in lowered:
                        score += 20
                    if 'sales increased' in lowered:
                        score += 12
                    if 'sales decreased' in lowered:
                        score += 12

                    # Causal wording is especially valuable for "why" questions.
                    if 'primarily due to' in lowered:
                        score += 15
                    elif 'due to' in lowered:
                        score += 10

                    if 'higher net sales' in lowered:
                        score += 10
                    if 'lower net sales' in lowered:
                        score += 10

                    if (
                        'revenue' in question_lower
                        or 'sales' in question_lower
                        or 'growth' in question_lower
                    ):
                        if 'net sales' in lowered:
                            score += 10
                        if 'revenue' in lowered:
                            score += 6

                    if (
                        'management' in section.lower()
                        or 'results of operations' in section.lower()
                    ):
                        score += 6

                    # These may contain similar causal phrases but do not explain
                    # revenue movement, so penalize them for revenue questions.
                    if (
                        'revenue' in question_lower
                        or 'sales' in question_lower
                        or 'growth' in question_lower
                    ):
                        if 'selling, general and administrative' in lowered:
                            score -= 25
                        if 'operating expense' in lowered:
                            score -= 20
                        if 'r&d expense' in lowered:
                            score -= 20
                        if 'research and development' in lowered:
                            score -= 20
                        if 'income tax' in lowered:
                            score -= 25
                        if 'effective tax rate' in lowered:
                            score -= 25
                        if 'gross margin' in lowered:
                            score -= 12

                    if 'financial statements' in section.lower():
                        score -= 2

                    if 'risk factors' in section.lower():
                        score -= 6

                    evidence_summaries.append(
                        (
                            score,
                            f"{sentence[:500]} [{citation_id}]",
                        )
                    )

        evidence_summaries.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        selected: list[str] = []
        seen: set[str] = set()

        for _, sentence in evidence_summaries:
            normalized = sentence.lower()[:120]

            if normalized in seen:
                continue

            seen.add(normalized)
            selected.append(sentence)

            if len(selected) >= 3:
                break

        if facts and facts != 'None':
            text = f"Demo mode: {facts}"

            if selected:
                text += (
                    "\n\nFiling evidence relevant to the explanation: "
                    + " ".join(selected)
                )
            else:
                text += (
                    "\n\nThe structured values above come from SEC XBRL "
                    "and/or deterministic calculations."
                )

            return GenerationResult(
                text=text,
                model='mock',
            )

        if not selected:
            return GenerationResult(
                'Insufficient evidence in the retrieved SEC filing '
                'to answer this confidently.',
                model='mock',
            )

        return GenerationResult(
            text='Demo mode: ' + ' '.join(selected),
            model='mock',
        )


class OpenAICompatibleProvider(LLMProvider):
    """Minimal vendor-neutral client for OpenAI-compatible `/chat/completions` APIs."""

    def __init__(self, settings: Settings):
        if not settings.llm_api_key:
            raise ValueError(
                'LLM_API_KEY is required for openai_compatible provider'
            )
        self.settings = settings

    def generate(self, system_prompt: str, user_prompt: str) -> GenerationResult:
        url = (
            self.settings.llm_base_url.rstrip('/')
            + '/chat/completions'
        )

        payload = {
            'model': self.settings.llm_model,
            'messages': [
                {
                    'role': 'system',
                    'content': system_prompt,
                },
                {
                    'role': 'user',
                    'content': user_prompt,
                },
            ],
            'temperature': self.settings.llm_temperature,
        }

        headers = {
            'Authorization': f'Bearer {self.settings.llm_api_key}',
            'Content-Type': 'application/json',
        }

        with httpx.Client(
            timeout=self.settings.llm_timeout_seconds
        ) as client:
            response = client.post(
                url,
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        text = data['choices'][0]['message']['content']
        usage = data.get('usage', {})

        return GenerationResult(
            text=text,
            input_tokens=int(
                usage.get('prompt_tokens', 0)
            ),
            output_tokens=int(
                usage.get('completion_tokens', 0)
            ),
            model=str(
                data.get(
                    'model',
                    self.settings.llm_model,
                )
            ),
        )


class AnswerGenerator:
    def __init__(
        self,
        settings: Settings | None = None,
        provider: LLMProvider | None = None,
    ):
        self.settings = settings or get_settings()

        if provider is not None:
            self.provider = provider
        elif (
            self.settings.demo_mode
            or self.settings.llm_provider == 'mock'
        ):
            self.provider = MockProvider()
        else:
            self.provider = OpenAICompatibleProvider(
                self.settings
            )

    def generate(
        self,
        question: str,
        evidence: str,
        structured_facts: str = '',
    ) -> GenerationResult:
        prompt = build_user_prompt(
            question,
            evidence,
            structured_facts,
        )

        return self.provider.generate(
            SYSTEM_PROMPT,
            prompt,
        )