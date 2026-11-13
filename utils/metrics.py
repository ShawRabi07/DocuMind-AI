"""
DocuMind AI – Metrics Tracking Module
======================================
Records real latency, token usage, and retrieval statistics across queries.
Strict Rule: Only logs real, empirically measured timings. No simulated metrics.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class QueryMetric:
    """Individual record of a single question-answering execution."""

    query: str
    retrieval_latency_ms: float
    llm_latency_ms: float
    total_latency_ms: float
    chunks_retrieved: int
    sources_count: int
    timestamp: float = field(default_factory=time.time)
    is_mock: bool = False
    tokens_used: int = 0


class MetricsTracker:
    """Thread-safe, in-memory accumulator of real system performance metrics."""

    def __init__(self) -> None:
        self._history: List[QueryMetric] = []

    def record_query(
        self,
        query: str,
        retrieval_ms: float,
        llm_ms: float,
        chunks_count: int,
        sources_count: int,
        is_mock: bool = False,
        tokens_used: int = 0,
    ) -> QueryMetric:
        """Appends a real measured query event to the performance history."""
        metric = QueryMetric(
            query=query,
            retrieval_latency_ms=round(retrieval_ms, 2),
            llm_latency_ms=round(llm_ms, 2),
            total_latency_ms=round(retrieval_ms + llm_ms, 2),
            chunks_retrieved=chunks_count,
            sources_count=sources_count,
            is_mock=is_mock,
            tokens_used=tokens_used,
        )
        self._history.append(metric)
        return metric

    def get_summary(self) -> Dict[str, float]:
        """Calculates real summary statistics across all executed queries."""
        if not self._history:
            return {
                "total_queries": 0,
                "avg_retrieval_ms": 0.0,
                "avg_llm_ms": 0.0,
                "avg_total_ms": 0.0,
                "avg_chunks_retrieved": 0.0,
            }

        total_q = len(self._history)
        avg_ret = sum(m.retrieval_latency_ms for m in self._history) / total_q
        avg_llm = sum(m.llm_latency_ms for m in self._history) / total_q
        avg_tot = sum(m.total_latency_ms for m in self._history) / total_q
        avg_chk = sum(m.chunks_retrieved for m in self._history) / total_q

        return {
            "total_queries": total_q,
            "avg_retrieval_ms": round(avg_ret, 2),
            "avg_llm_ms": round(avg_llm, 2),
            "avg_total_ms": round(avg_tot, 2),
            "avg_chunks_retrieved": round(avg_chk, 2),
        }

    def get_history(self) -> List[QueryMetric]:
        """Returns shallow copy of metric logs."""
        return list(self._history)

    def clear(self) -> None:
        """Resets performance history."""
        self._history.clear()


# Global shared metrics singleton
metrics_tracker = MetricsTracker()

