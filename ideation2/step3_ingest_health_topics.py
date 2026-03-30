"""
Step 3: Embed + Ingest into Elasticsearch (Unified Health Knowledge)
====================================================================
Takes outputs from step2 and:

1. Generates 768-dim dense vector embeddings (BioLord-2023-C)
   on the `semantic_text` field — trained on SNOMED-CT, MeSH, ICD-10
   medical ontologies for accurate symptom↔condition↔drug matching.

2. Bulk-indexes into Elasticsearch with:
   - BM25 text fields for keyword search
   - Dense vector field for semantic/KNN search
   - Keyword fields for exact filters (topic_type, body_systems)
   - Completion field for autocomplete
   - Custom health_analyzer with medical synonym expansion

3. Creates a symptom→condition map index for fast symptom lookup.

4. Creates a keyword alias map index for O(1) term resolution.

ES Indices created:
  neurohealth_health_topics   — full semantic + keyword docs (hybrid search)
  neurohealth_symptom_map     — symptom phrase → conditions list
  neurohealth_health_kwmap    — flat keyword → canonical name lookup

Run: python step3_ingest_health_topics.py
"""

import json
import os
import time
from datetime import datetime, timezone
from tqdm import tqdm
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────────────────────

ES_HOST = "https://127.0.0.1:9200"
ES_USER = "elastic"
ES_PASS = "dcJwpNq4m6wtDNpdbUhY"

EMBED_MODEL = "FremyCompany/BioLord-2023-C"  # Medical ontology-trained (SNOMED-CT, MeSH, ICD-10)
EMBED_DIMS = 768
BATCH_SIZE = 50

INPUT_SEMANTIC = "output/processed/health_topics_semantic.json"
INPUT_SYMPTOM_MAP = "output/processed/symptom_condition_map.json"
INPUT_KEYWORD_MAP = "output/processed/health_keyword_map.json"

INDEX_TOPICS = "neurohealth_health_topics"
INDEX_SYMPTOMS = "neurohealth_symptom_map"
INDEX_KWMAP = "neurohealth_health_kwmap"


# ── Index Mappings ───────────────────────────────────────────────────────────

TOPICS_MAPPING = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "analysis": {
            "analyzer": {
                "health_analyzer": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": [
                        "lowercase",
                        "asciifolding",
                        "health_synonyms",
                    ]
                }
            },
            "filter": {
                "health_synonyms": {
                    "type": "synonym",
                    "synonyms": [
                        # Symptom synonyms
                        "headache, cephalalgia, head pain, migraine",
                        "stomachache, stomach ache, abdominal pain, belly pain, tummy ache",
                        "chest pain, angina, chest tightness, thoracic pain",
                        "back pain, backache, lumbago, dorsalgia",
                        "sore throat, pharyngitis, throat pain",
                        "fever, pyrexia, high temperature, febrile",
                        "fatigue, tiredness, exhaustion, lethargy, malaise",
                        "nausea, queasiness, feeling sick, upset stomach",
                        "dizziness, vertigo, lightheadedness, giddiness",
                        "rash, skin rash, dermatitis, skin eruption",
                        "cough, coughing, tussis",
                        "breathlessness, shortness of breath, dyspnea, difficulty breathing",
                        "insomnia, sleeplessness, sleep disorder, difficulty sleeping",
                        "anxiety, anxiousness, nervousness, worry",
                        "depression, depressed, low mood, melancholy",
                        # Condition synonyms
                        "heart attack, myocardial infarction, cardiac arrest",
                        "stroke, cerebrovascular accident, brain attack",
                        "diabetes, diabetes mellitus, blood sugar disease",
                        "high blood pressure, hypertension, elevated bp",
                        "asthma, bronchial asthma, reactive airway",
                        "arthritis, joint inflammation, joint disease",
                        "flu, influenza, grippe",
                        "cold, common cold, upper respiratory infection, URI",
                        "uti, urinary tract infection, bladder infection",
                        "std, sexually transmitted disease, sti, sexually transmitted infection",
                        # Drug synonyms (kept from original)
                        "paracetamol, acetaminophen, tylenol, panadol",
                        "ibuprofen, advil, nurofen, motrin",
                        "aspirin, acetylsalicylic acid",
                        "metformin, glucophage",
                    ]
                }
            }
        }
    },
    "mappings": {
        "properties": {
            # ── Identity ──────────────────────────────────
            "doc_id":  {"type": "keyword"},
            "module":  {"type": "keyword"},
            "source":  {"type": "keyword"},

            # ── Topic name (BM25 + keyword + autocomplete) ──
            "topic_name": {
                "type": "text",
                "analyzer": "health_analyzer",
                "fields": {
                    "keyword": {"type": "keyword"},
                    "suggest": {
                        "type": "completion",
                        "analyzer": "simple",
                    }
                }
            },

            # ── Classification ────────────────────────────
            "topic_type": {"type": "keyword"},
            "body_systems": {"type": "keyword"},
            "group_names": {"type": "keyword"},
            "mesh_terms": {"type": "keyword"},

            # ── Synonyms & aliases ────────────────────────
            "also_called": {
                "type": "text",
                "analyzer": "health_analyzer",
                "fields": {
                    "keyword": {"type": "keyword"}
                }
            },
            "see_references": {
                "type": "text",
                "fields": {
                    "keyword": {"type": "keyword"}
                }
            },

            # ── Symptoms (critical for symptom matching) ──
            "symptoms": {
                "type": "text",
                "analyzer": "health_analyzer",
                "fields": {
                    "keyword": {"type": "keyword"}
                }
            },

            # ── Related topics ────────────────────────────
            "related_topic_names": {
                "type": "text",
                "fields": {
                    "keyword": {"type": "keyword"}
                }
            },

            # ── Semantic content (BM25 + embedding) ───────
            "keywords": {
                "type": "text",
                "analyzer": "health_analyzer",
                "fields": {
                    "keyword": {"type": "keyword"}
                }
            },
            "summary": {
                "type": "text",
                "analyzer": "health_analyzer",
            },
            "semantic_text": {
                "type": "text",
                "analyzer": "health_analyzer",
            },

            # ── Dense vector (KNN semantic search) ────────
            "embedding": {
                "type": "dense_vector",
                "dims": EMBED_DIMS,
                "index": True,
                "similarity": "cosine",
            },

            # ── Metadata ──────────────────────────────────
            "url":         {"type": "keyword"},
            "ingested_at": {"type": "date"},
        }
    }
}

SYMPTOM_MAP_MAPPING = {
    "settings": {"number_of_shards": 1, "number_of_replicas": 0},
    "mappings": {
        "properties": {
            "symptom": {
                "type": "text",
                "analyzer": "standard",
                "fields": {
                    "keyword": {"type": "keyword"}
                }
            },
            "conditions": {
                "type": "nested",
                "properties": {
                    "condition":    {"type": "keyword"},
                    "body_systems": {"type": "keyword"},
                    "topic_type":   {"type": "keyword"},
                    "url":          {"type": "keyword"},
                }
            },
            "condition_count": {"type": "integer"},
            "embedding": {
                "type": "dense_vector",
                "dims": EMBED_DIMS,
                "index": True,
                "similarity": "cosine",
            },
        }
    }
}

KWMAP_MAPPING = {
    "settings": {"number_of_shards": 1, "number_of_replicas": 0},
    "mappings": {
        "properties": {
            "keyword":        {"type": "keyword"},
            "canonical_name": {"type": "keyword"},
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


# ── ES Connection ────────────────────────────────────────────────────────────

def connect_es() -> Elasticsearch:
    """Connect to Elasticsearch with basic auth."""
    es = Elasticsearch(
        ES_HOST,
        basic_auth=(ES_USER, ES_PASS),
        verify_certs=False,
    )

    if not es.ping():
        print("ERROR: Cannot connect to Elasticsearch.")
        raise ConnectionError("Elasticsearch not reachable")

    info = es.info()
    print(f"Connected to ES: {info['version']['number']}")
    return es


# ── Index Setup ──────────────────────────────────────────────────────────────

def setup_indices(es: Elasticsearch):
    for idx, mapping in [
        (INDEX_TOPICS, TOPICS_MAPPING),
        (INDEX_SYMPTOMS, SYMPTOM_MAP_MAPPING),
        (INDEX_KWMAP, KWMAP_MAPPING),
    ]:
        if es.indices.exists(index=idx):
            es.indices.delete(index=idx)
        es.indices.create(index=idx, body=mapping)
        print(f"Created index: {idx}")


# ── Bulk Ingest: Health Topics ───────────────────────────────────────────────

def ingest_health_topics(es: Elasticsearch, docs: list[dict]):
    print(f"\nEmbedding + indexing {len(docs)} health topic documents...")
    total_ok = 0

    for batch_start in tqdm(range(0, len(docs), BATCH_SIZE)):
        batch = docs[batch_start: batch_start + BATCH_SIZE]

        texts = [d["semantic_text"] for d in batch]
        embeddings = embed_batch(texts)

        actions = []
        for doc, emb in zip(batch, embeddings):
            es_doc = {
                **doc,
                "embedding": emb,
                "ingested_at": datetime.now(timezone.utc).isoformat(),
            }
            actions.append({
                "_index": INDEX_TOPICS,
                "_id": doc["doc_id"],
                "_source": es_doc,
            })

        ok, errors = helpers.bulk(es, actions, raise_on_error=False)
        total_ok += ok
        if errors:
            print(f"  [WARN] {len(errors)} errors at batch {batch_start}")

    es.indices.refresh(index=INDEX_TOPICS)
    print(f"Indexed: {total_ok} health topics")


# ── Bulk Ingest: Symptom Map ────────────────────────────────────────────────

def ingest_symptom_map(es: Elasticsearch, symptom_map: dict):
    print(f"\nEmbedding + indexing {len(symptom_map)} symptom map entries...")
    entries = list(symptom_map.items())
    total_ok = 0

    for batch_start in tqdm(range(0, len(entries), BATCH_SIZE)):
        batch = entries[batch_start: batch_start + BATCH_SIZE]

        texts = [symptom for symptom, _ in batch]
        embeddings = embed_batch(texts)

        actions = []
        for (symptom, conditions), emb in zip(batch, embeddings):
            doc_id = f"sym_{hash(symptom) & 0xFFFFFFFF:08x}"
            actions.append({
                "_index": INDEX_SYMPTOMS,
                "_id": doc_id,
                "_source": {
                    "symptom": symptom,
                    "conditions": conditions,
                    "condition_count": len(conditions),
                    "embedding": emb,
                },
            })

        ok, errors = helpers.bulk(es, actions, raise_on_error=False)
        total_ok += ok

    es.indices.refresh(index=INDEX_SYMPTOMS)
    print(f"Indexed: {total_ok} symptom map entries")


# ── Bulk Ingest: Keyword Map ────────────────────────────────────────────────

def ingest_kwmap(es: Elasticsearch, kwmap: dict):
    print(f"\nIndexing {len(kwmap)} keyword map entries...")
    actions = []
    for keyword, canonical in kwmap.items():
        actions.append({
            "_index": INDEX_KWMAP,
            "_id": keyword,
            "_source": {
                "keyword": keyword,
                "canonical_name": canonical,
                "keyword_text": keyword,
            }
        })

    for i in range(0, len(actions), 1000):
        batch = actions[i:i + 1000]
        helpers.bulk(es, batch, raise_on_error=False)

    es.indices.refresh(index=INDEX_KWMAP)
    print(f"Indexed: {len(kwmap)} keyword map entries")


# ═══════════════════════════════════════════════════════════════════════════════
# QUERY INTERFACE — used by the conversational agent
# ═══════════════════════════════════════════════════════════════════════════════

def lookup_topic(es: Elasticsearch, user_term: str) -> str | None:
    """
    O(1) keyword map lookup.
    Resolves any alias/synonym to canonical topic name.
    e.g. "heart attack" → "Heart Attack"
    e.g. "high bp" → "High Blood Pressure"
    """
    try:
        result = es.get(index=INDEX_KWMAP, id=user_term.lower().strip())
        return result["_source"]["canonical_name"]
    except Exception:
        return None


def search_by_symptoms(es: Elasticsearch, symptoms: list[str],
                       n: int = 10) -> list[dict]:
    """
    Find conditions matching a list of user-reported symptoms.
    Uses both keyword match AND semantic similarity on the symptom map.

    e.g. search_by_symptoms(["headache", "fever", "stiff neck"])
         → [{condition: "Meningitis", ...}, {condition: "Flu", ...}]
    """
    query_text = ", ".join(symptoms)
    vec = embed_batch([query_text])[0]

    body = {
        "size": n,
        "query": {
            "bool": {
                "should": [
                    {"terms": {"symptom.keyword": symptoms, "boost": 3.0}},
                    {"match": {"symptom": {"query": query_text, "boost": 1.5}}},
                ],
                "minimum_should_match": 1,
            }
        },
        "knn": {
            "field": "embedding",
            "query_vector": vec,
            "k": n,
            "num_candidates": n * 10,
        },
        "_source": ["symptom", "conditions", "condition_count"],
    }

    hits = es.search(index=INDEX_SYMPTOMS, body=body)["hits"]["hits"]

    # Aggregate conditions across matched symptom entries
    condition_scores = {}
    for hit in hits:
        src = hit["_source"]
        for cond in src["conditions"]:
            name = cond["condition"]
            if name not in condition_scores:
                condition_scores[name] = {
                    "condition": name,
                    "body_systems": cond.get("body_systems", []),
                    "url": cond.get("url", ""),
                    "matched_symptoms": [],
                    "score": 0,
                }
            condition_scores[name]["matched_symptoms"].append(src["symptom"])
            condition_scores[name]["score"] += 1

    # Sort by number of matched symptoms
    ranked = sorted(condition_scores.values(), key=lambda x: -x["score"])
    return ranked[:n]


def search_health_topics(es: Elasticsearch, query: str, n: int = 5,
                         topic_type: str = None,
                         body_system: str = None) -> list[dict]:
    """
    Hybrid BM25 + semantic search across ALL health topics.
    Handles: typos, synonyms, colloquial terms, medical terms.

    e.g. "i have a bad headache and feel nauseous"
         → finds Migraine, Tension Headache, Concussion, etc.
    """
    vec = embed_batch([query])[0]

    filter_clause = []
    if topic_type:
        filter_clause.append({"term": {"topic_type": topic_type}})
    if body_system:
        filter_clause.append({"term": {"body_systems": body_system}})

    body = {
        "size": n,
        "query": {
            "bool": {
                "should": [
                    # Exact topic name match
                    {"term": {"topic_name.keyword": {"value": query, "boost": 5.0}}},
                    # BM25 across all text fields
                    {"multi_match": {
                        "query": query,
                        "fields": [
                            "topic_name^3",
                            "also_called^2.5",
                            "symptoms^2.5",
                            "keywords^1.5",
                            "summary^1.2",
                            "semantic_text",
                            "mesh_terms^1.5",
                        ],
                        "type": "best_fields",
                        "fuzziness": "AUTO",
                        "boost": 2.0,
                    }},
                    # Synonym-aware match via health_analyzer
                    {"match": {
                        "semantic_text": {
                            "query": query,
                            "analyzer": "health_analyzer",
                            "boost": 1.5,
                        }
                    }},
                    # Symptom field match (critical for symptom queries)
                    {"match": {
                        "symptoms": {
                            "query": query,
                            "analyzer": "health_analyzer",
                            "boost": 2.0,
                        }
                    }},
                ],
                "filter": filter_clause,
                "minimum_should_match": 1,
            }
        },
        "knn": {
            "field": "embedding",
            "query_vector": vec,
            "k": n,
            "num_candidates": n * 15,
            **({"filter": filter_clause} if filter_clause else {}),
        },
        "_source": [
            "topic_name", "topic_type", "also_called", "body_systems",
            "symptoms", "summary", "url", "related_topic_names",
        ],
    }

    hits = es.search(index=INDEX_TOPICS, body=body)["hits"]["hits"]
    return [
        {
            "topic_name":    h["_source"]["topic_name"],
            "topic_type":    h["_source"].get("topic_type", ""),
            "also_called":   h["_source"].get("also_called", [])[:3],
            "body_systems":  h["_source"].get("body_systems", []),
            "symptoms":      h["_source"].get("symptoms", [])[:5],
            "summary":       h["_source"].get("summary", "")[:300],
            "url":           h["_source"].get("url", ""),
            "related":       h["_source"].get("related_topic_names", [])[:3],
            "score":         round(h.get("_score") or 0, 4),
        }
        for h in hits
    ]


def autocomplete_topic(es: Elasticsearch, prefix: str, n: int = 5) -> list[str]:
    """Autocomplete health topic name as user types."""
    body = {
        "suggest": {
            "topic_suggest": {
                "prefix": prefix,
                "completion": {
                    "field": "topic_name.suggest",
                    "size": n,
                    "skip_duplicates": True,
                }
            }
        }
    }
    resp = es.search(index=INDEX_TOPICS, body=body)
    options = resp.get("suggest", {}).get("topic_suggest", [])
    return [o["text"] for o in options[0].get("options", [])] if options else []


def get_related_conditions(es: Elasticsearch, topic_name: str,
                           n: int = 5) -> list[dict]:
    """
    Find conditions related to a given topic.
    Useful for conversational follow-up exploration.
    """
    # First get the topic document
    body = {
        "size": 1,
        "query": {"term": {"topic_name.keyword": topic_name}},
        "_source": ["related_topic_names", "body_systems", "symptoms", "semantic_text"],
    }
    hits = es.search(index=INDEX_TOPICS, body=body)["hits"]["hits"]
    if not hits:
        return []

    src = hits[0]["_source"]
    related_names = src.get("related_topic_names", [])

    # Fetch related topics
    if related_names:
        body = {
            "size": n,
            "query": {"terms": {"topic_name.keyword": related_names}},
            "_source": ["topic_name", "topic_type", "body_systems", "symptoms", "url"],
        }
        related_hits = es.search(index=INDEX_TOPICS, body=body)["hits"]["hits"]
        return [h["_source"] for h in related_hits]

    # Fallback: semantic similarity
    return search_health_topics(es, src.get("semantic_text", topic_name)[:200], n=n)


# ── Demo ─────────────────────────────────────────────────────────────────────

def demo(es: Elasticsearch):
    print("\n" + "=" * 60)
    print("HEALTH TOPIC SEARCH DEMO")
    print("=" * 60)

    # 1. Keyword lookup
    print("\n--- Keyword Lookups ---")
    test_lookups = ["heart attack", "flu", "diabetes", "headache", "migraine"]
    for term in test_lookups:
        result = lookup_topic(es, term)
        print(f"  '{term}' → {result}")

    # 2. Symptom-based search (the core use case)
    print("\n--- Symptom-Based Condition Search ---")
    symptom_queries = [
        ["headache", "fever", "stiff neck"],
        ["chest pain", "shortness of breath"],
        ["fatigue", "weight gain", "hair loss"],
        ["nausea", "abdominal pain", "diarrhea"],
    ]
    for symptoms in symptom_queries:
        print(f"\n  Symptoms: {symptoms}")
        results = search_by_symptoms(es, symptoms, n=5)
        for r in results[:3]:
            matched = r["matched_symptoms"][:3]
            print(f"    → {r['condition']} (matched: {matched}, systems: {r['body_systems']})")

    # 3. Natural language health search
    print("\n--- Natural Language Search ---")
    nl_queries = [
        ("i have a bad headache and feel nauseous", None, None),
        ("my child has a fever and rash", None, None),
        ("difficulty breathing when exercising", None, None),
        ("feeling anxious and can't sleep", None, "Mental Health"),
        ("medicine for diabetes", "drug", None),
    ]
    for query, ttype, bsys in nl_queries:
        label = query
        if ttype:
            label += f" [type:{ttype}]"
        if bsys:
            label += f" [system:{bsys}]"
        print(f"\n  Query: '{label}'")
        results = search_health_topics(es, query, n=3, topic_type=ttype, body_system=bsys)
        for r in results:
            symptoms_str = f" (symptoms: {', '.join(r['symptoms'][:3])})" if r["symptoms"] else ""
            print(f"    [{r['topic_type']}] {r['topic_name']}{symptoms_str}")

    # 4. Autocomplete
    print("\n--- Autocomplete ---")
    for prefix in ["head", "diab", "asth", "can"]:
        suggestions = autocomplete_topic(es, prefix)
        print(f"  '{prefix}' → {suggestions}")

    # 5. Related conditions
    print("\n--- Related Conditions ---")
    for topic in ["Migraine", "Diabetes"]:
        related = get_related_conditions(es, topic, n=3)
        names = [r.get("topic_name", "?") for r in related]
        print(f"  '{topic}' → {names}")


# ── Main ─────────────────────────────────────────────────────────────────────

def run():
    print("=" * 60)
    print("Step 3: Embed + Ingest into Elasticsearch")
    print("=" * 60)

    es = connect_es()

    # Load data from step 2
    with open(INPUT_SEMANTIC, encoding="utf-8") as f:
        semantic_docs = json.load(f)
    with open(INPUT_SYMPTOM_MAP, encoding="utf-8") as f:
        symptom_map = json.load(f)
    with open(INPUT_KEYWORD_MAP, encoding="utf-8") as f:
        kwmap = json.load(f)

    print(f"Loaded: {len(semantic_docs)} topics, "
          f"{len(symptom_map)} symptom entries, "
          f"{len(kwmap)} keyword entries")

    # Create indices
    setup_indices(es)

    # Ingest everything
    ingest_health_topics(es, semantic_docs)
    ingest_symptom_map(es, symptom_map)
    ingest_kwmap(es, kwmap)

    # Demo
    demo(es)

    # Final stats
    print(f"\n{'=' * 60}")
    print("Ingestion Complete:")
    print(f"  {INDEX_TOPICS}  : {es.count(index=INDEX_TOPICS)['count']} docs")
    print(f"  {INDEX_SYMPTOMS} : {es.count(index=INDEX_SYMPTOMS)['count']} entries")
    print(f"  {INDEX_KWMAP}    : {es.count(index=INDEX_KWMAP)['count']} entries")


if __name__ == "__main__":
    run()
