"""
DocuMind AI – Main Application Entry Point
==========================================
Streamlit Multi-Document RAG Dashboard integrating:
Home Dashboard | Document Management | AI Chat Assistant | Analytics | Settings.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List
import streamlit as st

from config.settings import settings
from core.rag_pipeline import RAGPipeline
from components.sidebar import render_sidebar
from components.chat_interface import render_chat_interface
from components.analytics import render_analytics
from utils.helpers import clean_filename, format_bytes

# Configure Root Logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("DocuMindAI")

# Page Configuration
st.set_page_config(
    page_title="DocuMind AI – Multi-Document RAG Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Load Custom CSS
def load_css() -> None:
    css_path = Path(__file__).resolve().parent / "assets" / "style.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


load_css()


# Cache Pipeline Instance across Streamlit Reruns
@st.cache_resource(show_spinner="Initializing DocuMind AI Engine...")
def get_rag_pipeline() -> RAGPipeline:
    """Initializes and caches the persistent RAG pipeline in memory."""
    logger.info("Initializing cached RAGPipeline singleton...")
    return RAGPipeline()


# Retrieve Cached Pipeline
pipeline: RAGPipeline = get_rag_pipeline()

# Render Sidebar Navigation
sidebar_state = render_sidebar(pipeline)
current_page = sidebar_state["selected_page"]
active_top_k = sidebar_state["top_k"]
active_doc_id = sidebar_state["selected_doc_id"]
active_mock_mode = sidebar_state["use_mock_llm"]


# =============================================================================
# VIEW 1: HOME DASHBOARD
# =============================================================================
if current_page == "🏠 Home":
    st.markdown(
        """
        <div class="hero-header">
            <h1 class="hero-title">DocuMind AI</h1>
            <p class="hero-subtitle">Production-grade Multi-Document Retrieval-Augmented Generation (RAG) Assistant powered by Llama 3 and ChromaDB.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Overview Metrics Row
    stats = pipeline.get_document_statistics()
    health = pipeline.check_system_health()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Uploaded Documents</div>
                <div class="metric-value">{stats.get("total_documents", 0)}</div>
                <div class="metric-subtitle">Active indexed PDFs</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Indexed Chunks</div>
                <div class="metric-value">{stats.get("total_chunks", 0)}</div>
                <div class="metric-subtitle">Stored dense vectors</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        emb_short = settings.embedding_model_name.split("/")[-1]
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Embedding Model</div>
                <div class="metric-value" style="font-size: 1.15rem; padding-top: 0.4rem;">{emb_short}</div>
                <div class="metric-subtitle">{health.get("embedding_dimension", 384)} Dimensions</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        llm_label = "Mock (Test)" if active_mock_mode else settings.ollama_model
        status_label = "🟡 Testing" if active_mock_mode else ("🟢 Active" if health.get("ollama_running") else "🔴 Offline")
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">LLM Backend</div>
                <div class="metric-value" style="font-size: 1.25rem; padding-top: 0.35rem;">{llm_label}</div>
                <div class="metric-subtitle">{status_label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### 🚀 Quick Start Guide")
    q1, q2, q3 = st.columns(3)
    with q1:
        st.markdown(
            """
            #### 1. Upload Documents
            Navigate to **📁 My Documents** in the sidebar. Drop multiple PDF research papers, technical manuals, or textbooks.
            """
        )
    with q2:
        st.markdown(
            """
            #### 2. Ask Questions
            Head over to **💬 AI Assistant**. Ask complex questions across all documents or filter by a specific paper.
            """
        )
    with q3:
        st.markdown(
            """
            #### 3. Inspect Citations
            Review every source document, page number, and similarity score to verify answers without hallucinations.
            """
        )

    st.markdown("---")
    st.markdown("### 🏗️ Architecture Workflow")
    st.info(
        "**Ingestion Pipeline:** PyMuPDF (Text Extraction) ➔ RecursiveCharacterTextSplitter (Page-by-Page Chunking) ➔ "
        "Sentence Transformers (384-dim Vectors) ➔ ChromaDB (Persistent HNSW Vector Storage).\n\n"
        "**Inference Pipeline:** Query ➔ Vector Embedding ➔ Cosine Similarity Search ➔ Top-K Extraction ➔ "
        "Strict Grounding Prompt ➔ Llama 3 via Ollama ➔ Final Answer with Verifiable Source Citations."
    )


# =============================================================================
# VIEW 2: MY DOCUMENTS (DOCUMENT MANAGEMENT)
# =============================================================================
elif current_page == "📁 My Documents":
    st.markdown(
        """
        <div class="hero-header">
            <h1 class="hero-title">Document Management</h1>
            <p class="hero-subtitle">Upload, process, inspect, and manage multi-page PDF documents.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Upload Section
    st.subheader("📤 Upload New Documents")
    uploaded_files = st.file_uploader(
        "Upload one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload standard text-based PDFs up to 25 MB per file.",
    )

    if uploaded_files:
        st.write(f"Selected **{len(uploaded_files)}** file(s):")
        for u in uploaded_files:
            st.caption(f"• `{u.name}` ({format_bytes(u.size)})")

        if st.button("⚡ Index Uploaded Documents", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            saved_paths: List[Path] = []
            for i, uploaded_file in enumerate(uploaded_files):
                safe_name = clean_filename(uploaded_file.name)
                save_dest = settings.upload_dir / safe_name

                # Save file to upload directory
                with open(save_dest, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                saved_paths.append(save_dest)

            # Ingest through the RAG pipeline
            total_files = len(saved_paths)
            for idx, p in enumerate(saved_paths):
                status_text.text(f"Processing ({idx + 1}/{total_files}): {p.name}...")
                res = pipeline.ingest_document(p)

                if res["status"] == "success":
                    st.success(
                        f"✅ **{res['file_name']}**: Indexed {res['chunks_added']} chunks across {res['total_pages']} pages."
                    )
                elif res["status"] == "skipped":
                    st.warning(f"⚠️ **{res['file_name']}**: {res['message']}")
                else:
                    st.error(f"❌ **{res['file_name']}**: {res.get('error')}")

                progress_bar.progress((idx + 1) / total_files)

            status_text.text("Ingestion completed!")
            st.rerun()

    st.markdown("---")

    # Indexed Documents List & Deletion
    st.subheader("📚 Currently Indexed Documents")
    stats = pipeline.get_document_statistics()
    docs = stats.get("documents", [])

    if not docs:
        st.info("No documents are currently indexed in ChromaDB.")
    else:
        st.write(f"Total: **{len(docs)}** document(s) with **{stats.get('total_chunks', 0)}** total chunks.")

        for d in docs:
            c_doc, c_stats, c_action = st.columns([4, 3, 2])
            with c_doc:
                st.markdown(f"**📄 {d['source']}**")
                st.caption(f"Document ID: `{d['document_id']}`")
            with c_stats:
                st.write(f"**{d['chunk_count']}** chunks | **{d['total_pages']}** page(s)")
            with c_action:
                del_key = f"del_{d['document_id']}"
                if st.button("🗑️ Delete", key=del_key, type="secondary"):
                    deleted = pipeline.delete_document(d["document_id"])
                    st.toast(f"Deleted {d['source']} ({deleted} chunks removed).")
                    st.rerun()
            st.markdown("<hr style='margin: 0.5rem 0; opacity: 0.2;'>", unsafe_allow_html=True)


# =============================================================================
# VIEW 3: AI CHAT ASSISTANT
# =============================================================================
elif current_page == "💬 AI Assistant":
    render_chat_interface(
        pipeline=pipeline,
        top_k=active_top_k,
        document_id=active_doc_id,
        use_mock_llm=active_mock_mode,
    )


# =============================================================================
# VIEW 4: ANALYTICS
# =============================================================================
elif current_page == "📊 Analytics":
    render_analytics(pipeline)


# =============================================================================
# VIEW 5: SETTINGS
# =============================================================================
elif current_page == "⚙️ Settings":
    st.markdown(
        """
        <div class="hero-header">
            <h1 class="hero-title">System Settings & Configuration</h1>
            <p class="hero-subtitle">Inspect backend runtime configurations and hardware parameters.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    s1, s2 = st.columns(2)

    with s1:
        st.subheader("🤖 LLM Configuration")
        st.text_input("Ollama Base URL", value=settings.ollama_base_url, disabled=True)
        st.text_input("Ollama Model Name", value=settings.ollama_model, disabled=True)
        st.number_input("Max Output Tokens", value=settings.ollama_max_tokens, disabled=True)
        st.number_input("Temperature", value=settings.ollama_temperature, disabled=True)
        st.caption("Edit `.env` file to customize Ollama endpoints and model parameters.")

    with s2:
        st.subheader("🧩 Embedding & Vector Store")
        st.text_input("Embedding Model", value=settings.embedding_model_name, disabled=True)
        st.text_input("Vector Database Backend", value=settings.vector_store_backend.upper(), disabled=True)
        st.number_input("Chunk Size (Characters)", value=settings.chunk_size, disabled=True)
        st.number_input("Chunk Overlap (Characters)", value=settings.chunk_overlap, disabled=True)
        st.text_input("ChromaDB Directory", value=str(settings.chroma_persist_dir), disabled=True)

    st.markdown("---")
    st.subheader("🩺 Diagnostic Connection Test")
    if st.button("Run Health Probe", type="primary"):
        h = pipeline.check_system_health()
        st.json(h)

