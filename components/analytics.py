"""
DocuMind AI – Analytics Component
==================================
Renders empirical system performance charts, document vector distributions,
and latency telemetry using Plotly with interactive filters.
Strict Operational Rule: Only plots real, empirically measured values.
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
            <h1 class="hero-title">Telemetry & System Analytics</h1>
            <p class="hero-subtitle">Real-time empirical metrics measuring vector indexing density, HNSW retrieval latency, and model inference.</p>
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
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    with kpi1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Indexed Documents</div>
                <div class="metric-value">{total_docs}</div>
                <div class="metric-subtitle">Active verified PDFs</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Total Chunks</div>
                <div class="metric-value">{total_chunks}</div>
                <div class="metric-subtitle">Stored 384-dim vectors</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi3:
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

    with kpi4:
        avg_llm_display = (
            f"{perf_summary['avg_llm_ms']:.1f} ms" if total_queries > 0 else "N/A"
        )
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Avg LLM Latency</div>
                <div class="metric-value">{avg_llm_display}</div>
                <div class="metric-subtitle">Token generation time</div>
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
                    "Avg Chunks/Page": round(d["chunk_count"] / max(1, d["total_pages"]), 2),
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
                color_continuous_scale=["#0284c7", "#38bdf8", "#818cf8"],
                title="Vector Chunks Indexed per Document",
                labels={"Chunks": "Number of Chunks", "Document": "PDF Filename"},
                template="plotly_dark",
            )
            fig_chunks.update_layout(
                plot_bgcolor="rgba(15, 23, 42, 0.5)",
                paper_bgcolor="rgba(15, 23, 42, 0.5)",
                margin=dict(l=20, r=20, t=45, b=20),
                font=dict(family="Plus Jakarta Sans, sans-serif"),
            )
            st.plotly_chart(fig_chunks, use_container_width=True)

        with col_chart2:
            fig_pie = px.pie(
                df_docs,
                names="Document",
                values="Chunks",
                title="Index Proportion by Document",
                hole=0.5,
                color_discrete_sequence=["#38bdf8", "#818cf8", "#34d399", "#f472b6"],
                template="plotly_dark",
            )
            fig_pie.update_layout(
                plot_bgcolor="rgba(15, 23, 42, 0.5)",
                paper_bgcolor="rgba(15, 23, 42, 0.5)",
                margin=dict(l=20, r=20, t=45, b=20),
                font=dict(family="Plus Jakarta Sans, sans-serif"),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    else:
        st.info("ℹ️ No documents are currently indexed in ChromaDB. Upload PDFs in **My Documents** to view index distributions.")

    st.markdown("---")

    # Real Query Latency Section
    st.subheader("⚡ End-to-End Query Latency Telemetry")

    if history:
        # Build DataFrame from real measured query metrics
        data_records = []
        for idx, m in enumerate(history, 1):
            data_records.append(
                {
                    "Query #": f"#{idx}",
                    "Query": m.query[:40] + ("..." if len(m.query) > 40 else ""),
                    "Retrieval (ms)": m.retrieval_latency_ms,
                    "LLM Generation (ms)": m.llm_latency_ms,
                    "Total Latency (ms)": m.total_latency_ms,
                    "Chunks": m.chunks_retrieved,
                    "Mode": "Mock LLM" if m.is_mock else "Live Ollama",
                }
            )

        df_metrics = pd.DataFrame(data_records)

        # Plot: Stacked Latency Breakdown
        fig_lat = go.Figure()
        fig_lat.add_trace(
            go.Bar(
                x=df_metrics["Query #"],
                y=df_metrics["Retrieval (ms)"],
                name="Vector Retrieval (ms)",
                marker=dict(color="#38bdf8", opacity=0.9),
            )
        )
        fig_lat.add_trace(
            go.Bar(
                x=df_metrics["Query #"],
                y=df_metrics["LLM Generation (ms)"],
                name="LLM Synthesis (ms)",
                marker=dict(color="#818cf8", opacity=0.9),
            )
        )

        fig_lat.update_layout(
            barmode="stack",
            title="Empirical Latency Breakdown per Executed Query",
            xaxis_title="Query Sequence",
            yaxis_title="Time (Milliseconds)",
            template="plotly_dark",
            plot_bgcolor="rgba(15, 23, 42, 0.5)",
            paper_bgcolor="rgba(15, 23, 42, 0.5)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=60, b=20),
            font=dict(family="Plus Jakarta Sans, sans-serif"),
        )
        st.plotly_chart(fig_lat, use_container_width=True)

        # Query Audit Log Table with clear button
        st.markdown("##### 📋 Measured Query Audit Log")
        st.dataframe(
            df_metrics,
            use_container_width=True,
            hide_index=True,
        )

    else:
        st.info(
            "📊 **No query telemetry recorded yet.** "
            "Ask questions in the **AI Assistant** section to record empirical vector retrieval and inference timings."
        )
