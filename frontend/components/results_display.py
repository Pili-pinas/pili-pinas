"""Results display component for the Pili-Pinas Streamlit UI."""

import streamlit as st


def render_results(response: dict) -> None:
    """
    Render the RAG query response.

    Args:
        response: Dict from the /query API endpoint.
    """
    answer = response.get("answer", "")
    sources = response.get("sources", [])
    chunks_used = response.get("chunks_used", 0)

    # Answer
    st.markdown("### Answer")
    st.markdown(answer)

    # Metadata
    if chunks_used:
        st.caption(f"Generated from {chunks_used} source chunks.")

    # Sources
    if sources:
        st.markdown("---")
        st.markdown("### Sources")
        for i, src in enumerate(sources, 1):
            title = src.get("title", "Untitled")
            url = src.get("url", "")
            source_site = src.get("source", "")
            date = src.get("date", "")
            score = src.get("score", 0)

            with st.expander(f"{i}. {title}", expanded=(i == 1)):
                cols = st.columns([3, 1])
                with cols[0]:
                    if url:
                        st.markdown(f"[{url}]({url})")
                    st.caption(f"{source_site}  ·  {date}  ·  Relevance: {score:.0%}")
    else:
        st.info("No source documents were retrieved for this query.")


def render_error(message: str) -> None:
    st.error(f"Something went wrong: {message}")


def render_no_results() -> None:
    st.warning(
        "Hindi mahanap ang sagot sa aming database. / "
        "No relevant information found in our database. "
        "Try rephrasing your question or checking the source websites directly."
    )
