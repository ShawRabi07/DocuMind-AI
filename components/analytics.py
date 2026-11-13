"""
DocuMind AI – Analytics Component
==================================
Renders empirical system performance charts, document vector distributions,
and latency telemetry using Plotly.
Strict Operational Rule: Only plots real, empirically measured values. No simulated data.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core.rag_pipeline import RAGPipeline
from utils.metrics import metrics_tracker


def render_analytics(pipeline: RAGPipeline) -> None:
    """Renders the comprehensive real-time performance and document analytics dashboard."""
    st.markdown(
        """
        <div class="hero-header">
            <h1 class="hero-title">System Analytics & Telemetry</h1>
            <p class="hero-subtitle">Real-time empirical telemetry measuring vector indexing, retrieval latency, and model inference.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Fetch Real Statistics
    doc_stats = pipeline.get_document_statistics()
    perf_summary = metrics_tracker.get_summary()
    history = metrics_tracker.get_history()

    total_docs = doc_stats.get("total_documents", 0)
    total_chunks = doc_stats.get("total_chunks", 0)
    total_queries = perf_summary.get("total_queries", 0)

    # Top KPI Metrics Row
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

    with kpi_col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Total Documents</div>
                <div class="metric-value">{total_docs}</div>
                <div class="metric-subtitle">Active indexed PDFs</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi_col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Total Chunks</div>
                <div class="metric-value">{total_chunks}</div>
                <div class="metric-subtitle">Indexed vector embeddings</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi_col3:
        avg_ret_display = (
            f"{perf_summary['avg_retrieval_ms']:.1f} ms" if total_queries > 0 else "N/A"
        )
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Avg Retrieval Latency</div>
                <div class="metric-value">{avg_ret_display}</div>
                <div class="metric-subtitle">ChromaDB HNSW search</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi_col4:
        avg_llm_display = (
            f"{perf_summary['avg_llm_ms']:.1f} ms" if total_queries > 0 else "N/A"
        )
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Avg LLM Latency</div>
                <div class="metric-value">{avg_llm_display}</div>
                <div class="metric-subtitle">Llama 3 generation</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Document Chunk Distribution Section
    st.subheader("📚 Document Vector Distribution")
    doc_list = doc_stats.get("documents", [])

    if doc_list:
        df_docs = pd.DataFrame(
            [
                {
                    "Document": d["source"],
                    "Chunks": d["chunk_count"],
                    "Pages": d["total_pages"],
                }
                for d in doc_list
            ]
        )

        col_chart1, col_chart2 = st.columns([3, 2])

        with col_chart1:
            fig_chunks = px.bar(
                df_docs,
                x="Document",
                y="Chunks",
                color="Chunks",
                color_continuous_scale="Blues",
                title="Vector Chunks Indexed per Document",
                labels={"Chunks": "Number of Chunks", "Document": "PDF Filename"},
                template="plotly_dark",
            )
            fig_chunks.update_layout(
                plot_bgcolor="rgba(15, 23, 42, 0.4)",
                paper_bgcolor="rgba(15, 23, 42, 0.4)",
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_chunks, use_container_width=True)

        with col_chart2:
            fig_pie = px.pie(
                df_docs,
                names="Document",
                values="Chunks",
                title="Index Proportion by Document",
                hole=0.45,
                template="plotly_dark",
            )
            fig_pie.update_layout(
                plot_bgcolor="rgba(15, 23, 42, 0.4)",
                paper_bgcolor="rgba(15, 23, 42, 0.4)",
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    else:
        st.info("No documents are currently indexed in ChromaDB. Upload PDFs to view vector distribution.")

    st.markdown("---")

    # Real Query Latency Section
    st.subheader("⚡ Query Latency Telemetry")

    if history:
        # Build DataFrame from real measured query metrics
        data_records = []
        for idx, m in enumerate(history, 1):
            data_records.append(
                {
                    "Query #": idx,
                    "Query": m.query[:35] + ("..." if len(m.query) > 35 else ""),
                    "Retrieval Latency (ms)": m.retrieval_latency_ms,
                    "LLM Latency (ms)": m.llm_latency_ms,
                    "Total Latency (ms)": m.total_latency_ms,
                    "Chunks Retrieved": m.chunks_retrieved,
                    "Mode": "Mock" if m.is_mock else "Live Ollama",
                }
            )

        df_metrics = pd.DataFrame(data_records)

        # Plot 1: Stacked Latency Breakdown
        fig_lat = go.Figure()
        fig_lat.add_trace(
            go.Bar(
                x=df_metrics["Query #"],
                y=df_metrics["Retrieval Latency (ms)"],
                name="Vector Retrieval (ms)",
                marker_color="#38bdf8",
            )
        )
        fig_lat.add_trace(
            go.Bar(
                x=df_metrics["Query #"],
                y=df_metrics["LLM Latency (ms)"],
                name="LLM Generation (ms)",
                marker_color="#818cf8",
            )
        )

        fig_lat.update_layout(
            barmode="stack",
            title="End-to-End Latency Breakdown per Query",
            xaxis_title="Query Execution Sequence",
            yaxis_title="Time (Milliseconds)",
            template="plotly_dark",
            plot_bgcolor="rgba(15, 23, 42, 0.4)",
            paper_bgcolor="rgba(15, 23, 42, 0.4)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=60, b=20),
        )
        st.plotly_chart(fig_lat, use_container_width=True)

        # Table: Query Execution Logs
        st.markdown("##### 📋 Measured Query Audit Log")
        st.dataframe(
            df_metrics[
                [
                    "Query #",
                    "Query",
                    "Retrieval Latency (ms)",
                    "LLM Latency (ms)",
                    "Total Latency (ms)",
                    "Chunks Retrieved",
                    "Mode",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    else:
        st.info(
            "📊 **No query metrics recorded yet.** "
            "Ask questions in the **AI Assistant** section to record empirical latency measurements."
        )

