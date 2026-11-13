"""
DocuMind AI – Sidebar Component
================================
Renders the navigation menu, system status, active model info,
retrieval hyperparameter controls, and database summary statistics.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import streamlit as st

from config.settings import settings
from core.rag_pipeline import RAGPipeline


def render_sidebar(pipeline: RAGPipeline) -> Dict[str, Any]:
    """
    Renders the unified sidebar and returns user-selected navigation and filters.

    Args:
        pipeline: The active RAGPipeline instance.

    Returns:
        Dictionary containing:
        - "selected_page": chosen navigation section.
        - "top_k": configured retrieval chunk limit.
        - "selected_doc_id": document filter ID (or None for all documents).
        - "use_mock_llm": boolean whether mock mode is forced.
    """
    with st.sidebar:
        # App Branding Header
        st.markdown(
            """
            <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1.25rem;">
                <span style="font-size: 2rem;">🧠</span>
                <div>
                    <h2 style="margin: 0; font-size: 1.35rem; font-weight: 700; color: #f8fafc;">DocuMind AI</h2>
                    <p style="margin: 0; font-size: 0.75rem; color: #94a3b8;">Multi-Document RAG Assistant</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # Navigation
        st.markdown("### 🧭 Navigation")
        pages = [
            "🏠 Home",
            "📁 My Documents",
            "💬 AI Assistant",
            "📊 Analytics",
            "⚙️ Settings",
        ]
        selected_page = st.radio(
            "Select Section",
            pages,
            label_visibility="collapsed",
            index=0,
        )

        st.markdown("---")

        # System Health & Ollama Status
        st.markdown("### 🔌 System Health")
        health = pipeline.check_system_health()
        ollama_running = health.get("ollama_running", False)
        target_model_avail = health.get("target_model_available", False)

        # Mock Mode Toggle
        mock_mode_default = st.session_state.get("use_mock_llm", not ollama_running)
        use_mock_llm = st.toggle(
            "Use Mock LLM (Testing Mode)",
            value=mock_mode_default,
            help="Enables simulated LLM generation for offline testing when Ollama is not running.",
        )
        st.session_state["use_mock_llm"] = use_mock_llm

        if use_mock_llm:
            st.markdown(
                '<span class="badge-mock">🟡 Mock LLM Active (Offline Test)</span>',
                unsafe_allow_html=True,
            )
        elif ollama_running:
            if target_model_avail:
                st.markdown(
                    f'<span class="badge-online">🟢 Ollama Online ({health.get("target_model_name")})</span>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<span class="badge-mock">🟡 Model Missing ({health.get("target_model_name")})</span>',
                    unsafe_allow_html=True,
                )
                st.caption(f"Run `ollama pull {health.get('target_model_name')}`")
        else:
            st.markdown(
                '<span class="badge-offline">🔴 Ollama Offline</span>',
                unsafe_allow_html=True,
            )
            st.caption("Local service at port 11434 not detected.")

        st.markdown("---")

        # Retrieval Controls
        st.markdown("### 🔍 Retrieval Controls")
        current_top_k = st.session_state.get("top_k", settings.top_k)
        top_k = st.slider(
            "Top-K Chunks to Retrieve",
            min_value=1,
            max_value=10,
            value=current_top_k,
            help="Number of most semantically relevant document chunks to send to the LLM.",
        )
        st.session_state["top_k"] = top_k

        # Document Filter Dropdown
        stats = pipeline.get_document_statistics()
        doc_list = stats.get("documents", [])

        doc_options = {"All Documents (Default)": None}
        for doc in doc_list:
            label = f"{doc['source']} ({doc['chunk_count']} chunks)"
            doc_options[label] = doc["document_id"]

        selected_label = st.selectbox(
            "Filter by Document",
            options=list(doc_options.keys()),
            index=0,
            help="Restrict retrieval to a specific document or search across all indexed PDFs.",
        )
        selected_doc_id = doc_options[selected_label]

        st.markdown("---")

        # Database Quick Overview
        st.markdown("### 📚 Database Overview")
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.metric("Documents", stats.get("total_documents", 0))
        with col_d2:
            st.metric("Total Chunks", stats.get("total_chunks", 0))

        # Clear Chat Button
        st.markdown("---")
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state["messages"] = []
            st.rerun()

        # Footer info
        st.caption(f"Embedding: `{settings.embedding_model_name.split('/')[-1]}`")
        st.caption(f"Backend: `ChromaDB (Persistent)`")

    return {
        "selected_page": selected_page,
        "top_k": top_k,
        "selected_doc_id": selected_doc_id,
        "use_mock_llm": use_mock_llm,
    }

