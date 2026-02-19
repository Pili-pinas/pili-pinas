"""
Pili-Pinas Streamlit frontend.

Run:
    streamlit run frontend/app.py
"""

import os
import requests
import streamlit as st

from components.search_interface import render_search_form
from components.results_display import render_results, render_error, render_no_results

API_URL = os.getenv("PILI_PINAS_API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Pili-Pinas — Informed Filipino Voters",
    page_icon="🗳️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Styles ───────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0 0.5rem;
    }
    .main-header h1 {
        font-size: 2.5rem;
        color: #0038A8;  /* Philippine blue */
    }
    .tagline {
        text-align: center;
        color: #CE1126;  /* Philippine red */
        font-size: 1.1rem;
        margin-bottom: 1.5rem;
    }
    .disclaimer {
        font-size: 0.8rem;
        color: #888;
        text-align: center;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────

st.markdown('<div class="main-header"><h1>🗳️ Pili-Pinas</h1></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="tagline">Imformed Filipino voters. · '
    'AI summaries of politician records, laws, and election data with citations.</div>',
    unsafe_allow_html=True,
)

# ── Search ───────────────────────────────────────────────────────────────────

question, source_type, top_k = render_search_form()

search_clicked = st.button("Search 🔍", type="primary", use_container_width=False)

# ── Results ──────────────────────────────────────────────────────────────────

if search_clicked:
    if not question:
        st.warning("Please enter a question first.")
    else:
        with st.spinner("Searching Philippine government records..."):
            try:
                payload = {
                    "question": question,
                    "top_k": top_k,
                }
                if source_type:
                    payload["source_type"] = source_type

                resp = requests.post(f"{API_URL}/query", json=payload, timeout=60)
                resp.raise_for_status()
                data = resp.json()

                if data.get("chunks_used", 0) == 0:
                    render_no_results()
                else:
                    render_results(data)

            except requests.exceptions.ConnectionError:
                render_error(
                    f"Cannot connect to API at {API_URL}. "
                    "Make sure the backend is running: `uvicorn backend.src.api.main:app --reload`"
                )
            except requests.exceptions.Timeout:
                render_error("The request timed out. Please try again.")
            except requests.exceptions.HTTPError as e:
                render_error(str(e))
            except Exception as e:
                render_error(str(e))

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("About Pili-Pinas")
    st.markdown("""
    **Pili-Pinas** helps Filipino voters make informed decisions by summarizing:
    - Politician voting records and profiles
    - Philippine laws and Senate/House bills
    - SALN financial disclosures
    - News coverage and investigative reports

    All answers cite their sources so you can verify every claim.

    ---

    **Data Sources**
    - senate.gov.ph
    - congress.gov.ph
    - officialgazette.gov.ph
    - comelec.gov.ph
    - Rappler, Inquirer, PhilStar, GMA News
    - PCIJ, Transparency International PH

    ---

    **Tech Stack**
    - LangChain + ChromaDB (RAG)
    - Multilingual embeddings (Filipino + English)
    - Ollama / Claude Haiku (LLM)
    """)

    st.markdown("---")

    # API health check
    try:
        health = requests.get(f"{API_URL}/health", timeout=3)
        if health.ok:
            st.success("API: Online ✅")
            stats = requests.get(f"{API_URL}/stats", timeout=3).json()
            st.caption(f"Database: {stats.get('total_chunks', 0):,} document chunks")
        else:
            st.error("API: Offline ❌")
    except Exception:
        st.error("API: Offline ❌")
        st.caption(f"Expected at {API_URL}")

# ── Disclaimer ───────────────────────────────────────────────────────────────

st.markdown(
    '<div class="disclaimer">'
    "Pili-Pinas provides information for educational purposes only. "
    "It does not endorse any candidate or political party. "
    "Always verify claims with primary sources."
    "</div>",
    unsafe_allow_html=True,
)
