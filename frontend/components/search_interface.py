"""Search input component for the Pili-Pinas Streamlit UI."""

import streamlit as st


SOURCE_TYPE_OPTIONS = {
    "All Sources": None,
    "Senate/House Bills": "bill",
    "Laws": "law",
    "News Articles": "news",
    "Politician Profiles": "profile",
    "SALN Disclosures": "saln",
    "Election Data": "election",
}

EXAMPLE_QUERIES = [
    "Ano ang rekord ni Cynthia Villar sa Senate?",
    "What bills has Senator Padilla authored?",
    "Who voted against the anti-dynasty bill?",
    "Ano ang naipasa na batas tungkol sa kalikasan?",
    "What is the status of the Universal Health Care Act?",
    "Sino ang mga kandidato para senador sa 2025?",
]


def render_search_form() -> tuple[str, str | None, int]:
    """
    Render the search input form.

    Returns:
        (question, source_type_filter, top_k)
    """
    st.subheader("Ask a question about Philippine politics")

    # Example queries as quick chips
    with st.expander("Example questions", expanded=False):
        cols = st.columns(2)
        for i, example in enumerate(EXAMPLE_QUERIES):
            if cols[i % 2].button(example, key=f"example_{i}", use_container_width=True):
                st.session_state["prefill_question"] = example

    # Main question input
    default_q = st.session_state.pop("prefill_question", "")
    question = st.text_area(
        label="Your question",
        value=default_q,
        placeholder="e.g. What is the voting record of Senator Juan dela Cruz on infrastructure bills?",
        height=100,
        key="question_input",
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        source_label = st.selectbox(
            "Filter by source type",
            options=list(SOURCE_TYPE_OPTIONS.keys()),
            index=0,
        )
        source_type = SOURCE_TYPE_OPTIONS[source_label]

    with col2:
        top_k = st.slider("Results to retrieve", min_value=3, max_value=15, value=5, step=1)

    return question.strip(), source_type, top_k
