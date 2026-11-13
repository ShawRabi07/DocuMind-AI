"""
DocuMind AI – Conversational Chat Interface Component
======================================================
Renders the ChatGPT-style document Q&A conversation, interactive question chips,
verifiable source citations with visual relevance meters, and real latency metrics.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import streamlit as st

from core.llm import MockOllamaClient, OllamaClient
from core.rag_pipeline import RAGPipeline, RAGResponse


def render_chat_interface(
    pipeline: RAGPipeline,
    top_k: int = 5,
    document_id: Optional[str] = None,
    use_mock_llm: bool = False,
) -> None:
    """
    Renders the conversational AI assistant view.

    Args:
        pipeline: The active RAGPipeline instance.
        top_k: Number of chunks to retrieve.
        document_id: Optional single-document filter.
        use_mock_llm: Whether to use simulated LLM generation.
    """
    st.markdown(
        """
        <div class="hero-header">
            <h1 class="hero-title">AI Research Assistant</h1>
            <p class="hero-subtitle">Ask questions across your multi-page PDFs with transparent source grounding and page-by-page citations.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Active Scope & Filter Pill Banner
    stats = pipeline.get_document_statistics()
    doc_count = stats.get("total_documents", 0)
    chunk_count = stats.get("total_chunks", 0)

    scope_col1, scope_col2 = st.columns([3, 1])
    with scope_col1:
        if document_id:
            # Find doc name
            matched = [d["source"] for d in stats.get("documents", []) if d["document_id"] == document_id]
            doc_name = matched[0] if matched else document_id
            st.markdown(
                f'<span class="filter-active-pill">🎯 Filter: <b>{doc_name}</b> (Top-{top_k} Chunks)</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<span class="filter-active-pill">🌐 Global Scope: <b>All {doc_count} Documents</b> ({chunk_count} Chunks, Top-{top_k})</span>',
                unsafe_allow_html=True,
            )
    with scope_col2:
        if st.button("🔄 Reset Chat", type="secondary", use_container_width=True):
            st.session_state["messages"] = []
            st.rerun()

    # Initialize chat history in session state
    if "messages" not in st.session_state or not st.session_state["messages"]:
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "content": (
                    "👋 **Hello! I am DocuMind AI**, your document research assistant.\n\n"
                    "Ask me anything about your uploaded documents. I will retrieve the most "
                    "relevant excerpts and synthesize a factual answer with verifiable page citations."
                ),
                "citations": [],
                "retrieved_chunks": [],
            }
        ]

    # Interactive Question Suggestion Chips
    if doc_count > 0 and len(st.session_state["messages"]) <= 1:
        st.markdown("##### 💡 Suggested Questions (Click to Ask):")
        chip_col1, chip_col2 = st.columns(2)

        prompt_candidates = [
            "How does the Self-Attention mechanism compute attention weights?",
            "What is Shor's algorithm and why does it threaten RSA?",
            "Explain how Vision Transformers (ViT) process images.",
            "What is quantum entanglement and how does it affect qubits?",
        ]

        clicked_prompt = None
        with chip_col1:
            if st.button(f"🔍 {prompt_candidates[0]}", use_container_width=True):
                clicked_prompt = prompt_candidates[0]
            if st.button(f"⚛️ {prompt_candidates[1]}", use_container_width=True):
                clicked_prompt = prompt_candidates[1]
        with chip_col2:
            if st.button(f"🖼️ {prompt_candidates[2]}", use_container_width=True):
                clicked_prompt = prompt_candidates[2]
            if st.button(f"🌌 {prompt_candidates[3]}", use_container_width=True):
                clicked_prompt = prompt_candidates[3]

        if clicked_prompt:
            st.session_state["pending_query"] = clicked_prompt
            st.rerun()

    st.markdown("---")

    # Display Conversation History
    for idx, msg in enumerate(st.session_state["messages"]):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # Render Source Citations if present
            citations = msg.get("citations", [])
            if citations:
                st.markdown("##### 📌 Verifiable Source Citations:")
                for cit in citations:
                    score_pct = cit.similarity_score * 100
                    fill_pct = max(5, min(100, int(score_pct)))
                    st.markdown(
                        f"""
                        <div class="citation-card">
                            <div class="citation-header">
                                <span class="citation-pill">📄 {cit.source}</span>
                                <div>
                                    <span style="font-size: 0.8rem; color: #94a3b8; font-weight: 500;">
                                        Page <b>{cit.page_number}</b> (Chunk #{cit.chunk_index}) &bull; Relevance: <b>{score_pct:.1f}%</b>
                                    </span>
                                    <span class="citation-score-bar-bg">
                                        <div class="citation-score-bar-fill" style="width: {fill_pct}%;"></div>
                                    </span>
                                </div>
                            </div>
                            <div class="citation-excerpt">"{cit.excerpt}"</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            # Render Expandable Retrieved Context Chunks
            chunks = msg.get("retrieved_chunks", [])
            if chunks:
                with st.expander(f"🔍 Inspect All {len(chunks)} Retrieved Document Chunks"):
                    for c_idx, chk in enumerate(chunks, 1):
                        st.markdown(
                            f"**Chunk #{c_idx}** &mdash; `{chk.source}` (Page {chk.page_number}) | "
                            f"Cosine Distance: `{chk.distance:.4f}` | Score: `{chk.similarity_score * 100:.1f}%`"
                        )
                        st.code(chk.text, language="text")

            # Render Real Latency Tags
            total_lat = msg.get("total_latency_ms")
            if total_lat is not None and total_lat > 0:
                ret_lat = msg.get("retrieval_latency_ms", 0.0)
                llm_lat = msg.get("llm_latency_ms", 0.0)
                mock_label = " [Mock LLM]" if msg.get("is_mock") else ""
                st.markdown(
                    f"""
                    <div class="latency-tag">
                        <span class="latency-item">⚡ Retrieval: <b>{ret_lat:.1f} ms</b></span>
                        <span class="latency-item">🤖 Generation: <b>{llm_lat:.1f} ms</b>{mock_label}</span>
                        <span class="latency-item">⏱️ Total: <b>{total_lat:.1f} ms</b></span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # Check for Pending Query from Suggestion Chips
    user_query = st.chat_input("Ask a question about your uploaded documents...")
    if not user_query and "pending_query" in st.session_state:
        user_query = st.session_state.pop("pending_query")

    if user_query:
        # Display user message immediately
        st.session_state["messages"].append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Prepare conversational history (last 4 messages)
        history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state["messages"][-5:-1]
        ]

        # Configure LLM mode on the pipeline dynamically
        if use_mock_llm and not isinstance(pipeline.llm_client, MockOllamaClient):
            pipeline.llm_client = MockOllamaClient()
        elif not use_mock_llm and isinstance(pipeline.llm_client, MockOllamaClient):
            pipeline.llm_client = OllamaClient()

        # Generate Grounded RAG Response
        with st.chat_message("assistant"):
            with st.spinner("Searching document vectors & synthesizing grounded answer..."):
                response: RAGResponse = pipeline.ask_question(
                    query=user_query,
                    top_k=top_k,
                    document_id=document_id,
                    conversation_history=history,
                )

            st.markdown(response.answer)

            # Display citations
            if response.citations:
                st.markdown("##### 📌 Verifiable Source Citations:")
                for cit in response.citations:
                    score_pct = cit.similarity_score * 100
                    fill_pct = max(5, min(100, int(score_pct)))
                    st.markdown(
                        f"""
                        <div class="citation-card">
                            <div class="citation-header">
                                <span class="citation-pill">📄 {cit.source}</span>
                                <div>
                                    <span style="font-size: 0.8rem; color: #94a3b8; font-weight: 500;">
                                        Page <b>{cit.page_number}</b> (Chunk #{cit.chunk_index}) &bull; Relevance: <b>{score_pct:.1f}%</b>
                                    </span>
                                    <span class="citation-score-bar-bg">
                                        <div class="citation-score-bar-fill" style="width: {fill_pct}%;"></div>
                                    </span>
                                </div>
                            </div>
                            <div class="citation-excerpt">"{cit.excerpt}"</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            # Display expandable retrieved chunks
            if response.retrieved_chunks:
                with st.expander(
                    f"🔍 Inspect All {len(response.retrieved_chunks)} Retrieved Document Chunks"
                ):
                    for c_idx, chk in enumerate(response.retrieved_chunks, 1):
                        st.markdown(
                            f"**Chunk #{c_idx}** &mdash; `{chk.source}` (Page {chk.page_number}) | "
                            f"Cosine Distance: `{chk.distance:.4f}` | Score: `{chk.similarity_score * 100:.1f}%`"
                        )
                        st.code(chk.text, language="text")

            # Display latency breakdown
            mock_label = " [Mock LLM]" if response.is_mock else ""
            st.markdown(
                f"""
                <div class="latency-tag">
                    <span class="latency-item">⚡ Retrieval: <b>{response.retrieval_latency_ms:.1f} ms</b></span>
                    <span class="latency-item">🤖 Generation: <b>{response.llm_latency_ms:.1f} ms</b>{mock_label}</span>
                    <span class="latency-item">⏱️ Total: <b>{response.total_latency_ms:.1f} ms</b></span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Append assistant response to session state
        st.session_state["messages"].append(
            {
                "role": "assistant",
                "content": response.answer,
                "citations": response.citations,
                "retrieved_chunks": response.retrieved_chunks,
                "retrieval_latency_ms": response.retrieval_latency_ms,
                "llm_latency_ms": response.llm_latency_ms,
                "total_latency_ms": response.total_latency_ms,
                "model_used": response.model_used,
                "is_mock": response.is_mock,
            }
        )
