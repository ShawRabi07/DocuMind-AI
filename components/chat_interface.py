"""
DocuMind AI – Conversational Chat Interface Component
======================================================
Renders the ChatGPT-style document Q&A conversation, displays verifiable
source citations, provides expandable retrieved context previews, and shows real latency.
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
            <h1 class="hero-title">AI Document Assistant</h1>
            <p class="hero-subtitle">Ask complex questions across your uploaded documents with precise page-level citations.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "content": (
                    "Hello! I am DocuMind AI, your multi-document research assistant. "
                    "Upload PDFs in the **My Documents** section, and ask me any question. "
                    "I will answer strictly using facts verified in your documents with page citations."
                ),
                "citations": [],
                "retrieved_chunks": [],
            }
        ]

    # Quick prompt suggestion chips if chat is empty or fresh
    stats = pipeline.get_document_statistics()
    doc_count = stats.get("total_documents", 0)

    if doc_count == 0:
        st.info(
            "💡 **No documents indexed yet.** Go to **My Documents** in the sidebar to upload your PDFs.",
            icon="📄",
        )

    # Display Conversation History
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # Render Source Citations if present
            citations = msg.get("citations", [])
            if citations:
                st.markdown("##### 📌 Verifiable Source Citations:")
                for cit in citations:
                    score_pct = f"{cit.similarity_score * 100:.1f}%"
                    st.markdown(
                        f"""
                        <div class="citation-card">
                            <div class="citation-header">
                                <span>📄 {cit.source} — Page {cit.page_number} (Chunk #{cit.chunk_index})</span>
                                <span class="citation-score">Relevance: {score_pct}</span>
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
                    for idx, chk in enumerate(chunks, 1):
                        st.markdown(
                            f"**Chunk #{idx}** &mdash; `{chk.source}` (Page {chk.page_number}) | "
                            f"Cosine Distance: `{chk.distance:.4f}`"
                        )
                        st.code(chk.text, language="text")

            # Render Latency Tags
            total_lat = msg.get("total_latency_ms")
            if total_lat is not None and total_lat > 0:
                ret_lat = msg.get("retrieval_latency_ms", 0.0)
                llm_lat = msg.get("llm_latency_ms", 0.0)
                mock_label = " [Mock LLM]" if msg.get("is_mock") else ""
                st.markdown(
                    f"""
                    <div class="latency-tag">
                        <span>⚡ Retrieval: <b>{ret_lat:.1f} ms</b></span>
                        <span>🤖 Generation: <b>{llm_lat:.1f} ms</b>{mock_label}</span>
                        <span>⏱️ Total: <b>{total_lat:.1f} ms</b></span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # Chat Input Box
    user_query = st.chat_input("Ask a question about your uploaded documents...")

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
                    score_pct = f"{cit.similarity_score * 100:.1f}%"
                    st.markdown(
                        f"""
                        <div class="citation-card">
                            <div class="citation-header">
                                <span>📄 {cit.source} — Page {cit.page_number} (Chunk #{cit.chunk_index})</span>
                                <span class="citation-score">Relevance: {score_pct}</span>
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
                    for idx, chk in enumerate(response.retrieved_chunks, 1):
                        st.markdown(
                            f"**Chunk #{idx}** &mdash; `{chk.source}` (Page {chk.page_number}) | "
                            f"Cosine Distance: `{chk.distance:.4f}`"
                        )
                        st.code(chk.text, language="text")

            # Display latency breakdown
            mock_label = " [Mock LLM]" if response.is_mock else ""
            st.markdown(
                f"""
                <div class="latency-tag">
                    <span>⚡ Retrieval: <b>{response.retrieval_latency_ms:.1f} ms</b></span>
                    <span>🤖 Generation: <b>{response.llm_latency_ms:.1f} ms</b>{mock_label}</span>
                    <span>⏱️ Total: <b>{response.total_latency_ms:.1f} ms</b></span>
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

