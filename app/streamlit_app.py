from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_settings

settings = get_settings()
st.set_page_config(page_title='SEC Filing Intelligence', page_icon='📊', layout='wide')

st.title('SEC Filing Research & Financial Intelligence Copilot')
st.caption('Citation-grounded SEC filing research + XBRL financial intelligence')

with st.sidebar:
    st.header('Research settings')
    ticker = st.text_input('Ticker', value='AAPL').strip().upper()
    form = st.selectbox('Filing type', ['10-K', '10-Q'])
    filing_date = st.text_input('Filing date', placeholder='latest (leave blank)')
    st.number_input('Retrieved candidates', min_value=1, max_value=50, value=settings.retrieval_top_k, disabled=True)
    st.toggle('Reranker enabled', value=settings.reranker_enabled, disabled=True)
    st.text_input('Model/provider', value=f'{settings.llm_provider} / {settings.llm_model}', disabled=True)
    st.number_input('Temperature', value=settings.llm_temperature, disabled=True)
    if settings.demo_mode:
        st.info('DEMO MODE is enabled. Real SEC data is used; generation may use the local mock provider.')

question = st.text_area('Ask a filing or financial question', value="What was Apple's revenue growth in the latest fiscal year?", height=110)
run = st.button('Research', type='primary')

if run:
    if not ticker or not question.strip():
        st.error('Ticker and question are required.')
        st.stop()
    payload = {'ticker': ticker, 'form': form, 'question': question.strip(), 'filing_date': filing_date or None}
    try:
        with st.spinner('Retrieving SEC evidence...'):
            response = httpx.post(f'{settings.api_base_url}/query', json=payload, timeout=300.0)
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPError as exc:
        detail = ''
        if getattr(exc, 'response', None) is not None:
            try:
                detail = exc.response.json().get('detail', '')
            except Exception:
                detail = exc.response.text
        st.error(f'API request failed: {detail or exc}')
        st.stop()

    st.subheader('Answer')
    st.write(data['answer'])
    if data.get('caveats'):
        for caveat in data['caveats']:
            st.warning(caveat)

    st.subheader('KPI table')
    kpis = data.get('kpis', [])
    if kpis:
        table = pd.DataFrame(kpis).rename(columns={
            'name': 'Metric', 'value': 'Current', 'previous_value': 'Previous',
            'change': 'Change', 'source': 'Source', 'unit': 'Unit', 'period': 'Period',
        })
        st.dataframe(table, use_container_width=True, hide_index=True)
    else:
        st.caption('No structured KPI was needed for this query.')

    st.subheader('Sources')
    citations = data.get('citations', [])
    if citations:
        for cite in citations:
            with st.container(border=True):
                st.markdown(f"**[{cite['citation_id']}] {cite.get('form', '')} — {cite.get('section', 'Document')}**")
                st.caption(f"Filed: {cite.get('filing_date', '')} · Accession: {cite.get('accession_number', '')}")
                st.write(cite.get('snippet', ''))
                if cite.get('source_url'):
                    st.link_button('Open SEC filing', cite['source_url'])
    else:
        st.caption('This answer used structured SEC XBRL facts rather than filing passages.')

    st.subheader('Retrieved passages')
    for i, passage in enumerate(data.get('retrieved_passages', []), start=1):
        with st.expander(f"Evidence {i}: {passage.get('section', 'Document')}"):
            st.write(passage.get('text', ''))
            st.caption(passage.get('source_url', ''))

    st.subheader('Performance')
    timings = data.get('timings_ms', {})
    cols = st.columns(4)
    cols[0].metric('Retrieval', f"{timings.get('retrieval', 0):.1f} ms")
    cols[1].metric('Reranking', f"{timings.get('reranking', 0):.1f} ms")
    cols[2].metric('Generation', f"{timings.get('generation', 0):.1f} ms")
    cols[3].metric('Total API', f"{data.get('latency_ms', 0):.1f} ms")
    usage = data.get('token_usage', {})
    st.caption(
        f"Route: {data.get('route', '')} · Chunks: {len(data.get('retrieved_passages', []))} · "
        f"Tokens: in={usage.get('input', 0)}, out={usage.get('output', 0)} · "
        f"Estimated cost: {data.get('estimated_cost') if data.get('estimated_cost') is not None else 'not configured'}"
    )
