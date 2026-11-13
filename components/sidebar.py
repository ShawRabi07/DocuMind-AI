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
            <div style="display: flex; align-items: center; gap: 0.85rem; padding: 0.5rem 0 1.25rem 0;">
                <div style="background: linear-gradient(135deg, #38bdf8, #6366f1); padding: 0.65rem; border-radius: 12px; box-shadow: 0 4px 12px rgba(56, 189, 248, 0.3);">
                    <span style="font-size: 1.6rem; display: block; line-height: 1;">🧠</span>
                </div>
                <div>
                    <h2 style="margin: 0; font-size: 1.35rem; font-weight: 800; color: #f8fafc; letter-spacing: -0.02em;">DocuMind AI</h2>
                    <p style="margin: 0; font-size: 0.75rem; color: #94a3b8; font-weight: 500;">Multi-Document RAG</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Navigation
        st.markdown("##### 🧭 Navigation")
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
        st.markdown("##### 🔌 System Status")
        health = pipeline.check_system_health()
        ollama_running = health.get("ollama_running", False)
        target_model_avail = health.get("target_model_available", False)

        # Mock Mode Toggle
        mock_mode_default = st.session_state.get("use_mock_llm", not ollama_running)
        use_mock_llm = st.toggle(
            "Mock LLM (Offline Test)",
            value=mock_mode_default,
            help="Enables simulated LLM generation for testing without an active Ollama process.",
        )
        st.session_state["use_mock_llm"] = use_mock_llm

        if use_mock_llm:
            st.markdown(
                '<div style="margin-bottom: 0.5rem;"><span class="badge-mock">Mock Mode Active</span></div>',
                unsafe_allow_html=True,
            )
        elif ollama_running:
            if target_model_avail:
                st.markdown(
                    f'<div style="margin-bottom: 0.5rem;"><span class="badge-online">Ollama Online ({health.get("target_model_name")})</span></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div style="margin-bottom: 0.5rem;"><span class="badge-mock">Model Missing ({health.get("target_model_name")})</span></div>',
                    unsafe_allow_html=True,
                )
                st.caption(f"Run `ollama pull {health.get('target_model_name')}`")
        else:
            st.markdown(
                '<div style="margin-bottom: 0.5rem;"><span class="badge-offline">Ollama Offline</span></div>',
                unsafe_allow_html=True,
            )
            st.caption("Service not detected on port 11434.")

        st.markdown("---")

        # Retrieval Controls
        st.markdown("##### 🔍 Retrieval Settings")
        current_top_k = st.session_state.get("top_k", settings.top_k)
        top_k = st.slider(
            "Top-K Excerpts",
            min_value=1,
            max_value=10,
            value=current_top_k,
            help="Maximum number of relevant chunks retrieved per question.",
        )
        st.session_state["top_k"] = top_k

        # Document Filter Dropdown
        stats = pipeline.get_document_statistics()
        doc_list = stats.get("documents", [])

        doc_options = {"All Documents (Global)": None}
        for doc in doc_list:
            label = f"{doc['source']} ({doc['chunk_count']} chunks)"
            doc_options[label] = doc["document_id"]

        selected_label = st.selectbox(
            "Target Document Scope",
            options=list(doc_options.keys()),
            index=0,
            help="Filter search to a specific document or search across the entire library.",
        )
        selected_doc_id = doc_options[selected_label]

        st.markdown("---")

        # Database Quick Overview
        st.markdown("##### 📚 Database Summary")
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.metric("Documents", stats.get("total_documents", 0))
        with col_d2:
            st.metric("Total Chunks", stats.get("total_chunks", 0))

        # Clear Chat Button
        st.markdown("---")
        if st.button("🗑️ Clear Chat History", use_container_width=True, type="secondary"):
            st.session_state["messages"] = []
            st.rerun()

        # Footer info
        st.caption(f"Vector Space: `384-dim (MiniLM-L6-v2)`")
        st.caption(f"Database: `ChromaDB HNSW`")

    return {
        "selected_page": selected_page,
        "top_k": top_k,
        "selected_doc_id": selected_doc_id,
        "use_mock_llm": use_mock_llm,
    }
