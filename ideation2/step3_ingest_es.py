"""
Step 3: Embed + Ingest into Elasticsearch
==========================================
Takes the semantic JSON from step2 and:

1. Generates dense vector embeddings (384-dim, all-MiniLM-L6-v2)
   for the `semantic_text` field of each drug document

2. Bulk-indexes into Elasticsearch with:
   - BM25 text fields: drug_name, brand_names, keywords, uses, summary
   - Dense vector field: embedding (for semantic/KNN search)
   - Keyword fields: drug_class, mesh_terms (for exact filter)
   - Completion field: drug_name.suggest (for autocomplete)

3. Also ingests the keyword_map as a separate lightweight index
   for instant O(1) drug name lookup from any alias/brand name

ES Indices created:
  neurohealth_medications    — full semantic + keyword docs
  neurohealth_drug_kwmap     — flat keyword → canonical name lookup

Run: python step3_ingest_es.py
"""

import json
import os
import time
from tqdm import tqdm
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer

ES_HOST     = "http://localhost:9200"
EMBED_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE  = 50    # docs per bulk request

INPUT_SEMANTIC = "output/processed/drugs_semantic.json"
INPUT_KWMAP    = "output/processed/drugs_keyword_map.json"

INDEX_MEDS  = "neurohealth_medications"
INDEX_KWMAP = "neurohealth_drug_kwmap"

os.makedirs("output/es_ready", exist_ok=True)

# ── Index mappings ────────────────────────────────────────────────────────────

MEDICATIONS_MAPPING = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "analysis": {
            "analyzer": {
                "drug_analyzer": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": [
                        "lowercase",
                        "asciifolding",
                        "drug_synonyms",
                    ]
                }
            },
            "filter": {
                "drug_synonyms": {
                    "type": "synonym",
                    "synonyms": [
                        # Generic ↔ brand synonyms (common ones)
                        "paracetamol, acetaminophen, tylenol, panadol",
                        "ibuprofen, advil, nurofen, brufen, motrin",
                        "aspirin, acetylsalicylic acid, disprin",
                        "metformin, glucophage, fortamet, glumetza",
                        "atorvastatin, lipitor",
                        "omeprazole, prilosec, losec",
                        "salbutamol, albuterol, ventolin, proventil",
                        "amoxicillin, amoxil, trimox",
                        "sertraline, zoloft",
                        "fluoxetine, prozac",
                        "amlodipine, norvasc",
                        "lisinopril, zestril, prinivil",
                        "levothyroxine, synthroid, eltroxin",
                        "cetirizine, zyrtec, reactine",
                        "loratadine, claritin",
                        "pantoprazole, protonix, pantoloc",
                        "azithromycin, zithromax, z-pak",
                        "ciprofloxacin, cipro",
                        "warfarin, coumadin",
                        "clopidogrel, plavix",
                    ]
                }
            }
        }
    },
    "mappings": {
        "properties": {
            # ── Identity ──────────────────────────────────
            "doc_id":      {"type": "keyword"},
            "module":      {"type": "keyword"},
            "source":      {"type": "keyword"},

            # ── Drug name fields (BM25 + keyword + autocomplete) ──
            "drug_name": {
                "type": "text",
                "analyzer": "drug_analyzer",
                "fields": {
                    "keyword":  {"type": "keyword"},   # exact filter
                    "suggest":  {                       # autocomplete
                        "type": "completion",
                        "analyzer": "simple",
                    }
                }
            },
            "brand_names": {
                "type": "text",
                "analyzer": "drug_analyzer",
                "fields": {
                    "keyword": {
                        "type": "keyword",
                        "normalizer": "lowercase",
                    }
                }
            },
            "generic_name": {
                "type": "text",
                "analyzer": "drug_analyzer",
            },

            # ── Classification ────────────────────────────
            "drug_class": {
                "type": "keyword",
                "fields": {
                    "text": {"type": "text", "analyzer": "drug_analyzer"}
                }
            },
            "mesh_terms":  {"type": "keyword"},
            "group_names": {"type": "keyword"},

            # ── Semantic content (BM25 text search) ───────
            "uses": {
                "type": "text",
                "analyzer": "drug_analyzer",
            },
            "keywords": {
                "type": "text",
                "analyzer": "drug_analyzer",
                "fields": {
                    "keyword": {"type": "keyword"}
                }
            },
            "summary": {
                "type": "text",
                "analyzer": "drug_analyzer",
            },
            "semantic_text": {
                "type": "text",
                "analyzer": "drug_analyzer",
                "comment": "Combined text for both BM25 and embedding generation"
            },

            # ── Dense vector (KNN semantic search) ────────
            "embedding": {
                "type": "dense_vector",
                "dims": 384,
                "index": True,
                "similarity": "cosine",
            },

            # ── Metadata ──────────────────────────────────
            "url":          {"type": "keyword"},
            "ingested_at":  {"type": "date"},
        }
    }
}

KWMAP_MAPPING = {
    "settings": {"number_of_shards": 1, "number_of_replicas": 0},
    "mappings": {
        "properties": {
            "keyword":        {"type": "keyword"},   # exact lookup key
            "canonical_name": {"type": "keyword"},   # resolved drug name
            "keyword_text": {
                "type": "text",
                "analyzer": "standard",
            }
        }
    }
}


# ── Embedding ─────────────────────────────────────────────────────────────────

_model = None
def get_model():
    global _model
    if _model is None:
        print(f"Loading embedding model: {EMBED_MODEL}")
        _model = SentenceTransformer(EMBED_MODEL)
    return _model

def embed_batch(texts: list[str]) -> list[list[float]]:
    return get_model().encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=32,
    ).tolist()


# ── ES setup ──────────────────────────────────────────────────────────────────

def setup_indices(es: Elasticsearch):
    for idx, mapping in [(INDEX_MEDS, MEDICATIONS_MAPPING),
                         (INDEX_KWMAP, KWMAP_MAPPING)]:
        if es.indices.exists(index=idx):
            es.indices.delete(index=idx)
        es.indices.create(index=idx, body=mapping)
        print(f"Created index: {idx}")


# ── Bulk ingest medications ───────────────────────────────────────────────────

def ingest_medications(es: Elasticsearch, docs: list[dict]):
    print(f"\nEmbedding + indexing {len(docs)} drug documents...")
    total_ok = 0

    for batch_start in tqdm(range(0, len(docs), BATCH_SIZE)):
        batch = docs[batch_start: batch_start + BATCH_SIZE]

        # Generate embeddings for the batch
        texts     = [d["semantic_text"] for d in batch]
        embeddings = embed_batch(texts)

        from datetime import datetime
        actions = []
        for doc, emb in zip(batch, embeddings):
            es_doc = {**doc, "embedding": emb,
                      "ingested_at": datetime.utcnow().isoformat()}
            actions.append({
                "_index": INDEX_MEDS,
                "_id":    doc["doc_id"],
                "_source": es_doc,
            })

        ok, errors = helpers.bulk(es, actions, raise_on_error=False)
        total_ok  += ok
        if errors:
            print(f"  [WARN] {len(errors)} errors in batch starting at {batch_start}")

    es.indices.refresh(index=INDEX_MEDS)
    print(f"Indexed: {total_ok} medications")


def ingest_kwmap(es: Elasticsearch, kwmap: dict):
    """Index keyword map as flat docs for O(1) alias lookup."""
    print(f"\nIndexing {len(kwmap)} keyword map entries...")
    actions = []
    for keyword, canonical in kwmap.items():
        actions.append({
            "_index": INDEX_KWMAP,
            "_id":    keyword,   # keyword IS the doc id — instant lookup
            "_source": {
                "keyword":        keyword,
                "canonical_name": canonical,
                "keyword_text":   keyword,
            }
        })

    # Bulk in one shot (keyword map entries are tiny)
    for i in range(0, len(actions), 1000):
        batch = actions[i:i+1000]
        ok, _ = helpers.bulk(es, batch, raise_on_error=False)

    es.indices.refresh(index=INDEX_KWMAP)
    print(f"Indexed: {len(kwmap)} keyword map entries")


# ── Query interface ───────────────────────────────────────────────────────────

def lookup_drug(es: Elasticsearch, user_term: str) -> str | None:
    """
    Fast O(1) keyword map lookup.
    Resolves any alias/brand name to canonical drug name.
    e.g. "glucophage" → "Metformin"
    """
    try:
        result = es.get(index=INDEX_KWMAP, id=user_term.lower().strip())
        return result["_source"]["canonical_name"]
    except Exception:
        return None


def search_drugs(es: Elasticsearch, query: str, n: int = 5,
                 drug_class: str = None) -> list[dict]:
    """
    Hybrid BM25 + semantic search for drugs.
    Handles: typos, brand names, generic names, use cases.

    e.g. "medicine for high blood pressure" → finds antihypertensives
    e.g. "glucophage" → finds Metformin
    e.g. "antibiotcs for throat" → finds amoxicillin etc. (typo handled)
    """
    vec = embed_batch([query])[0]

    filter_clause = []
    if drug_class:
        filter_clause.append({"term": {"drug_class": drug_class}})

    body = {
        "size": n,
        "query": {
            "bool": {
                "should": [
                    # Exact drug name match (highest priority)
                    {"term":  {"drug_name.keyword": {"value": query, "boost": 5.0}}},
                    # BM25 on all text fields
                    {"multi_match": {
                        "query":  query,
                        "fields": [
                            "drug_name^3",
                            "brand_names^2.5",
                            "generic_name^2",
                            "keywords^1.5",
                            "uses^1.2",
                            "semantic_text",
                            "summary",
                        ],
                        "type": "best_fields",
                        "fuzziness": "AUTO",    # handles typos
                        "boost": 2.0,
                    }},
                    # Synonym expansion (glucophage → metformin via analyzer)
                    {"match": {
                        "semantic_text": {
                            "query":    query,
                            "analyzer": "drug_analyzer",
                            "boost":    1.5,
                        }
                    }},
                ],
                "filter": filter_clause,
                "minimum_should_match": 1,
            }
        },
        # KNN semantic vector search
        "knn": {
            "field":          "embedding",
            "query_vector":   vec,
            "k":              n,
            "num_candidates": n * 15,
            **({"filter": filter_clause} if filter_clause else {}),
        },
        # RRF fusion of BM25 + KNN
        "rank": {
            "rrf": {"window_size": 100, "rank_constant": 20}
        },
        "_source": [
            "drug_name", "brand_names", "drug_class",
            "uses", "summary", "url", "keywords",
        ],
    }

    hits = es.search(index=INDEX_MEDS, body=body)["hits"]["hits"]
    return [
        {
            "drug_name":   h["_source"]["drug_name"],
            "brand_names": h["_source"].get("brand_names", [])[:3],
            "drug_class":  h["_source"].get("drug_class", ""),
            "uses":        h["_source"].get("uses", [])[:3],
            "summary":     h["_source"].get("summary", "")[:200],
            "url":         h["_source"].get("url", ""),
            "score":       round(h.get("_score") or 0, 4),
        }
        for h in hits
    ]


def autocomplete_drug(es: Elasticsearch, prefix: str, n: int = 5) -> list[str]:
    """Autocomplete drug name as user types."""
    body = {
        "suggest": {
            "drug_suggest": {
                "prefix": prefix,
                "completion": {
                    "field": "drug_name.suggest",
                    "size": n,
                    "skip_duplicates": True,
                }
            }
        }
    }
    resp = es.search(index=INDEX_MEDS, body=body)
    options = resp.get("suggest", {}).get("drug_suggest", [])
    return [o["text"] for o in options[0].get("options", [])] if options else []


# ── Demo ──────────────────────────────────────────────────────────────────────

def demo(es: Elasticsearch):
    print("\n=== Drug Search Demo ===\n")

    test_queries = [
        ("medicine for type 2 diabetes",    None),
        ("glucophage",                       None),   # brand name lookup
        ("antibiotcs for chest infection",   None),   # typo
        ("blood pressure tablet",            None),
        ("painkiller",                       "Analgesic / Pain Relief"),
        ("inhaler for asthma",               None),
    ]

    for query, drug_class in test_queries:
        print(f"Query: '{query}'" + (f" [class: {drug_class}]" if drug_class else ""))

        # First try fast keyword map lookup
        canonical = lookup_drug(es, query)
        if canonical:
            print(f"  [KWMAP exact] → {canonical}")

        # Then do full hybrid search
        results = search_drugs(es, query, n=3, drug_class=drug_class)
        for r in results:
            brands = f" ({', '.join(r['brand_names'])})" if r["brand_names"] else ""
            print(f"  [{r['drug_class']}] {r['drug_name']}{brands}")
            if r["uses"]:
                print(f"    Uses: {', '.join(r['uses'][:2])}")
        print()

    # Autocomplete demo
    print("Autocomplete 'met' →", autocomplete_drug(es, "met"))
    print("Autocomplete 'ami' →", autocomplete_drug(es, "ami"))


def run():
    es = Elasticsearch(ES_HOST)
    if not es.ping():
        print("ERROR: Elasticsearch not running.")
        print("Start: docker run -d --name es-neurohealth -p 9200:9200 "
              "-e 'discovery.type=single-node' "
              "-e 'xpack.security.enabled=false' "
              "docker.elastic.co/elasticsearch/elasticsearch:8.13.0")
        return

    with open(INPUT_SEMANTIC, encoding="utf-8") as f:
        semantic_docs = json.load(f)
    with open(INPUT_KWMAP, encoding="utf-8") as f:
        kwmap = json.load(f)

    setup_indices(es)
    ingest_medications(es, semantic_docs)
    ingest_kwmap(es, kwmap)
    demo(es)

    print(f"\n=== Ingestion complete ===")
    print(f"  neurohealth_medications : {es.count(index=INDEX_MEDS)['count']} docs")
    print(f"  neurohealth_drug_kwmap  : {es.count(index=INDEX_KWMAP)['count']} entries")


if __name__ == "__main__":
    run()
