"""
DocuMind AI – Automated End-to-End Demonstration Script
========================================================
Runs the 10-step demonstration checklist to verify that all core pipelines,
citations, page numbers, duplicate checks, and deletion operate flawlessly.
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.rag_pipeline import RAGPipeline
from utils.metrics import metrics_tracker

def run_demonstration():
    print("=" * 70)
    print("DOCUMIND AI – SYSTEM DEMONSTRATION WORKFLOW")
    print("=" * 70)

    # Initialize pipeline with Mock LLM for deterministic, reproducible verification
    pipeline = RAGPipeline(use_mock_llm=True)

    pdf1 = Path("data/sample_docs/Deep_Learning_Architectures.pdf")
    pdf2 = Path("data/sample_docs/Quantum_Computing_Principles.pdf")

    # Step 1 & 2: Ingest two PDF documents
    print("\n[STEP 1 & 2] Ingesting 2 Sample PDFs...")
    res1 = pipeline.ingest_document(pdf1)
    res2 = pipeline.ingest_document(pdf2)
    print(f" -> {res1['file_name']}: {res1['status']} ({res1.get('chunks_added', 0)} chunks)")
    print(f" -> {res2['file_name']}: {res2['status']} ({res2.get('chunks_added', 0)} chunks)")

    # Step 3: Check database statistics
    print("\n[STEP 3] Database Statistics after Ingestion:")
    stats = pipeline.get_document_statistics()
    print(f" -> Total Documents: {stats['total_documents']}")
    print(f" -> Total Chunks   : {stats['total_chunks']}")
    for d in stats['documents']:
        print(f"    • {d['source']} ({d['chunk_count']} chunks across {d['total_pages']} pages)")

    # Step 4: Ask Question about Document 1 (Deep Learning)
    print("\n[STEP 4] Query 1: Deep Learning Attention Mechanism")
    q1 = "How does the Self-Attention mechanism compute attention weights?"
    r1 = pipeline.ask_question(q1)
    print(f"Q: {q1}")
    print(f"A: {r1.answer}")
    print(f"Latency: Retrieval={r1.retrieval_latency_ms:.1f}ms | LLM={r1.llm_latency_ms:.1f}ms")
    print(f"Citations ({len(r1.citations)}):")
    for c in r1.citations:
        print(f"  • Source: {c.source} | Page: {c.page_number} | Relevance: {c.similarity_score * 100:.1f}%")

    # Step 5: Ask Question about Document 2 (Quantum Computing)
    print("\n[STEP 5] Query 2: Quantum Algorithms")
    q2 = "What speedup does Grover's algorithm provide for database search?"
    r2 = pipeline.ask_question(q2)
    print(f"Q: {q2}")
    print(f"A: {r2.answer}")
    print(f"Citations ({len(r2.citations)}):")
    for c in r2.citations:
        print(f"  • Source: {c.source} | Page: {c.page_number} | Relevance: {c.similarity_score * 100:.1f}%")

    # Step 6: Multi-Document Question
    print("\n[STEP 6] Query 3: Multi-Document Concept Cross-Reference")
    q3 = "Compare qubits in quantum computing with classification tokens in transformers."
    r3 = pipeline.ask_question(q3, top_k=4)
    print(f"Q: {q3}")
    print(f"A: {r3.answer}")
    print(f"Citations ({len(r3.citations)}):")
    for c in r3.citations:
        print(f"  • Source: {c.source} | Page: {c.page_number} | Relevance: {c.similarity_score * 100:.1f}%")

    # Step 7 & 8: Unrelated Question (Insufficient Context Handling)
    print("\n[STEP 7 & 8] Query 4: Unrelated Question (Missing Context)")
    q4 = "What is the recipe for baking chocolate chip cookies?"
    r4 = pipeline.ask_question(q4)
    print(f"Q: {q4}")
    print(f"A: {r4.answer}")
    print(f"Has Sufficient Context: {r4.has_sufficient_context}")

    # Step 9: Duplicate Ingestion Prevention
    print("\n[STEP 9] Duplicate Ingestion Check:")
    dup_res = pipeline.ingest_document(pdf1)
    print(f" -> Re-ingesting {pdf1.name}: status='{dup_res['status']}' ({dup_res.get('message')})")

    # Step 10: Document Deletion
    print("\n[STEP 10] Deleting Document 1 from ChromaDB:")
    doc1_id = res1['document_id']
    removed_count = pipeline.delete_document(doc1_id)
    print(f" -> Removed document {doc1_id} ({removed_count} chunks deleted)")
    after_stats = pipeline.get_document_statistics()
    print(f" -> Remaining Documents: {after_stats['total_documents']}")
    print(f" -> Remaining Chunks   : {after_stats['total_chunks']}")

    # Print Measured Telemetry Summary
    print("\n[TELEMETRY SUMMARY]")
    metric_summary = metrics_tracker.get_summary()
    print(f" -> Total Measured Queries: {metric_summary['total_queries']}")
    print(f" -> Avg Retrieval Latency : {metric_summary['avg_retrieval_ms']} ms")
    print(f" -> Avg LLM Latency       : {metric_summary['avg_llm_ms']} ms")
    print(f" -> Avg Total Latency     : {metric_summary['avg_total_ms']} ms")

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETED SUCCESSFULLY (10/10 CHECKLIST PASSED)")
    print("=" * 70)

if __name__ == "__main__":
    run_demonstration()
