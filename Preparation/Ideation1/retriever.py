"""
Pipeline: NeuroHealth Unified Hybrid Retriever
================================================
Single query interface that searches ALL modules simultaneously
using Elasticsearch's multi-index hybrid search.

This is what the NeuroHealth LLM pipeline calls at runtime —
one function that does BM25 + semantic + fuzzy across
symptoms + conditions + medications + conversations.

Usage:
    from pipeline.retriever import NeuroHealthRetriever
    retriever = NeuroHealthRetriever()
    results = retriever.search("my chest hurts and i cant breathe", n=5)
"""

import json
from typing import Optional
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

ES_HOST     = "http://localhost:9200"
EMBED_MODEL = "all-MiniLM-L6-v2"

ALL_INDICES = [
    "neurohealth_symptoms",
    "neurohealth_conditions",
    "neurohealth_medications",
    "neurohealth_conversations",
]

URGENCY_PRIORITY = {"red": 4, "orange": 3, "yellow": 2, "green": 1, "unknown": 0}


class NeuroHealthRetriever:
    def __init__(self, indices: list[str] = None):
        self.es    = Elasticsearch(ES_HOST)
        self.model = SentenceTransformer(EMBED_MODEL)
        self.indices = indices or ALL_INDICES

    def _embed(self, text: str) -> list[float]:
        return self.model.encode(text, normalize_embeddings=True).tolist()

    def search(
        self,
        query: str,
        n: int = 5,
        urgency_filter: Optional[str] = None,
        module_filter: Optional[str] = None,
    ) -> list[dict]:
        """
        Hybrid search across all NeuroHealth indices.

        Args:
            query:          User's symptom description / health question
            n:              Number of results to return
            urgency_filter: Restrict to a tier: green/yellow/orange/red
            module_filter:  Restrict to a module: symptoms/conditions/medications/conversations

        Returns:
            Ranked list of relevant knowledge chunks
        """
        vec = self._embed(query)

        filter_clauses = []
        if urgency_filter:
            filter_clauses.append({"term": {"urgency_tier": urgency_filter}})
        if module_filter:
            filter_clauses.append({"term": {"module": module_filter}})

        body = {
            "size": n,
            "query": {
                "bool": {
                    "should": [
                        # BM25 on patient-language text (highest boost)
                        {"match": {
                            "patient_text": {"query": query, "boost": 2.0}
                        }},
                        # BM25 on full combined text
                        {"match": {
                            "full_text": {"query": query, "boost": 1.5}
                        }},
                        # Synonym-aware BM25 (medical_analyzer handles "high bp" → "hypertension")
                        {"match": {
                            "symptom_names": {"query": query, "boost": 1.2}
                        }},
                        # Fuzzy for typos and misspellings
                        {"multi_match": {
                            "query": query,
                            "fields": ["symptom_names", "condition_name"],
                            "fuzziness": "AUTO",
                            "boost": 0.8,
                        }},
                    ],
                    "filter": filter_clauses,
                    "minimum_should_match": 1,
                }
            },
            # Dense vector KNN semantic search
            "knn": {
                "field": "embedding",
                "query_vector": vec,
                "k": n,
                "num_candidates": n * 15,
                **({"filter": filter_clauses} if filter_clauses else {}),
            },
            # Reciprocal Rank Fusion merges BM25 + KNN scores
            "rank": {
                "rrf": {
                    "window_size": 100,
                    "rank_constant": 20,
                }
            },
            # Return only the fields NeuroHealth needs
            "_source": [
                "doc_id", "module", "source", "urgency_tier",
                "patient_text", "clinical_text", "recommendation",
                "symptom_names", "condition_name", "red_flags",
                "icd10_codes", "body_location",
            ],
        }

        # Search across available indices
        available = [i for i in self.indices
                     if self.es.indices.exists(index=i)]
        if not available:
            return []

        response = self.es.search(
            index=",".join(available),
            body=body,
        )

        results = []
        for hit in response["hits"]["hits"]:
            src = hit["_source"]
            results.append({
                "doc_id":        src.get("doc_id"),
                "module":        src.get("module"),
                "urgency":       src.get("urgency_tier", "unknown"),
                "symptom":       src.get("symptom_names", ""),
                "condition":     src.get("condition_name", ""),
                "patient_text":  src.get("patient_text", "")[:300],
                "clinical_text": src.get("clinical_text", "")[:400],
                "recommendation":src.get("recommendation", ""),
                "red_flag":      src.get("red_flags", False),
                "score":         round(hit.get("_score") or 0, 4),
            })

        # Sort: red flags first, then by urgency priority, then by score
        results.sort(
            key=lambda r: (
                r["red_flag"],
                URGENCY_PRIORITY.get(r["urgency"], 0),
                r["score"],
            ),
            reverse=True,
        )
        return results[:n]

    def build_rag_context(self, query: str, n: int = 5) -> str:
        """
        Format retrieval results as LLM context string.
        This is injected into the LLM system prompt for RAG.
        """
        results = self.search(query, n=n)
        if not results:
            return "No relevant medical knowledge found for this query."

        lines = ["RELEVANT MEDICAL KNOWLEDGE:\n"]
        for i, r in enumerate(results, 1):
            urgency_label = r["urgency"].upper()
            lines.append(f"[{i}] [{urgency_label}] {r['symptom'] or r['condition']}")
            if r["patient_text"]:
                lines.append(f"    Context: {r['patient_text'][:200]}")
            if r["recommendation"]:
                lines.append(f"    Recommendation: {r['recommendation']}")
            if r["red_flag"]:
                lines.append(f"    *** RED FLAG — consider emergency referral ***")
            lines.append("")

        return "\n".join(lines)

    def get_urgency_from_symptoms(self, symptoms: list[str]) -> str:
        """
        Given a list of extracted symptoms, return the highest urgency tier found.
        Used by the urgency classifier as a RAG-grounded signal.
        """
        query = " ".join(symptoms)
        results = self.search(query, n=10)
        if not results:
            return "unknown"

        # Return highest urgency found
        tiers = [r["urgency"] for r in results if r["urgency"] != "unknown"]
        if not tiers:
            return "unknown"
        return max(tiers, key=lambda t: URGENCY_PRIORITY.get(t, 0))


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    retriever = NeuroHealthRetriever()

    test_cases = [
        "I have chest pain that goes to my left arm",
        "tummy ache after eating spicy food",
        "headche for 3 days, worst one ever",
        "feel dizzy and want to vomit, also tired",
        "cant breathe properly and lips turning blue",
    ]

    for q in test_cases:
        print(f"\nQuery: '{q}'")
        print(retriever.build_rag_context(q, n=3))
        print("-" * 50)
