"""
Module 1: Symptom Extraction & Elasticsearch Indexing
=======================================================
This is the FIRST module to build — start here.

What it does:
  1. Pulls symptom data from MedlinePlus Search API (keyword-based)
  2. Pulls structured symptom-condition pairs from DDXPlus (HuggingFace)
  3. Extracts symptoms using spaCy medical NER + rule-based patterns
  4. Maps symptoms to: ICD-10 codes, body locations, synonyms, urgency hints
  5. Indexes everything into Elasticsearch with full-text + vector fields

Why symptoms first:
  Every other module (conditions, medications, conversations) references symptoms.
  Building this first gives you the vocabulary the rest of the pipeline uses.

Run:
  # Start Elasticsearch first (Docker):
  docker run -d --name es-neurohealth -p 9200:9200 \
    -e "discovery.type=single-node" \
    -e "xpack.security.enabled=false" \
    docker.elastic.co/elasticsearch/elasticsearch:8.13.0

  python modules/module1_symptoms.py

Output: Elasticsearch index "neurohealth_symptoms" populated
"""

import json
import time
import requests
import hashlib
from datetime import datetime
from typing import Optional
from tqdm import tqdm

try:
    from elasticsearch import Elasticsearch, helpers
    from sentence_transformers import SentenceTransformer
    import spacy
except ImportError:
    print("Install: pip install elasticsearch sentence-transformers spacy")
    print("         python -m spacy download en_core_web_sm")
    raise

ES_HOST  = "http://localhost:9200"
INDEX    = "neurohealth_symptoms"
EMBED_MODEL = "all-MiniLM-L6-v2"   # 384 dims, fast, free

# ── Symptom knowledge: canonical name → metadata ─────────────────────────────

SYMPTOM_CATALOG = {
    # Format: "canonical_name": {icd10, body_location, synonyms, urgency_hint, category}
    "chest pain": {
        "icd10": "R07.9",
        "body_location": "chest",
        "synonyms": ["chest tightness", "chest pressure", "chest discomfort",
                     "breast pain", "thoracic pain", "my chest hurts"],
        "urgency_hint": "red",
        "category": "cardiovascular",
        "red_flag": True,
    },
    "shortness of breath": {
        "icd10": "R06.0",
        "body_location": "chest",
        "synonyms": ["breathlessness", "dyspnea", "difficulty breathing",
                     "can't breathe", "breathless", "winded"],
        "urgency_hint": "orange",
        "category": "respiratory",
        "red_flag": True,
    },
    "headache": {
        "icd10": "R51",
        "body_location": "head",
        "synonyms": ["head pain", "head ache", "migraine", "cephalalgia",
                     "head is pounding", "my head hurts"],
        "urgency_hint": "yellow",
        "category": "neurological",
        "red_flag": False,
    },
    "abdominal pain": {
        "icd10": "R10.9",
        "body_location": "abdomen",
        "synonyms": ["stomach pain", "tummy ache", "belly pain", "stomach ache",
                     "stomach cramps", "abdominal cramps", "gut pain"],
        "urgency_hint": "yellow",
        "category": "gastrointestinal",
        "red_flag": False,
    },
    "fever": {
        "icd10": "R50.9",
        "body_location": "systemic",
        "synonyms": ["high temperature", "pyrexia", "febrile", "temperature",
                     "feeling hot", "burning up", "elevated temperature"],
        "urgency_hint": "yellow",
        "category": "general",
        "red_flag": False,
    },
    "dizziness": {
        "icd10": "R42",
        "body_location": "head",
        "synonyms": ["dizzy", "lightheaded", "vertigo", "spinning sensation",
                     "feeling faint", "unsteady", "balance problems"],
        "urgency_hint": "yellow",
        "category": "neurological",
        "red_flag": False,
    },
    "fatigue": {
        "icd10": "R53.83",
        "body_location": "systemic",
        "synonyms": ["tiredness", "exhaustion", "lethargy", "weakness",
                     "lack of energy", "feeling run down", "no energy"],
        "urgency_hint": "green",
        "category": "general",
        "red_flag": False,
    },
    "nausea": {
        "icd10": "R11.0",
        "body_location": "abdomen",
        "synonyms": ["feeling sick", "queasy", "wanting to vomit", "sick feeling",
                     "upset stomach", "feel like vomiting"],
        "urgency_hint": "green",
        "category": "gastrointestinal",
        "red_flag": False,
    },
    "vomiting": {
        "icd10": "R11.1",
        "body_location": "abdomen",
        "synonyms": ["throwing up", "being sick", "emesis", "retching",
                     "puking", "vomit"],
        "urgency_hint": "yellow",
        "category": "gastrointestinal",
        "red_flag": False,
    },
    "back pain": {
        "icd10": "M54.5",
        "body_location": "back",
        "synonyms": ["lower back pain", "backache", "back ache", "lumbar pain",
                     "spine pain", "my back hurts", "back is killing me"],
        "urgency_hint": "yellow",
        "category": "musculoskeletal",
        "red_flag": False,
    },
    "rash": {
        "icd10": "R21",
        "body_location": "skin",
        "synonyms": ["skin rash", "hives", "urticaria", "skin eruption",
                     "spots on skin", "red patches", "itchy rash"],
        "urgency_hint": "yellow",
        "category": "dermatological",
        "red_flag": False,
    },
    "sudden severe headache": {
        "icd10": "R51",
        "body_location": "head",
        "synonyms": ["worst headache of my life", "thunderclap headache",
                     "sudden head pain", "explosive headache"],
        "urgency_hint": "red",
        "category": "neurological",
        "red_flag": True,
    },
    "chest pain radiating to arm": {
        "icd10": "R07.9",
        "body_location": "chest",
        "synonyms": ["pain in chest and arm", "chest and left arm pain",
                     "arm pain with chest pain", "jaw and chest pain"],
        "urgency_hint": "red",
        "category": "cardiovascular",
        "red_flag": True,
    },
    "facial drooping": {
        "icd10": "R29.810",
        "body_location": "face",
        "synonyms": ["face drooping", "droopy face", "one side of face drooping",
                     "face looks uneven", "facial weakness"],
        "urgency_hint": "red",
        "category": "neurological",
        "red_flag": True,
    },
    "joint pain": {
        "icd10": "M25.50",
        "body_location": "joints",
        "synonyms": ["joint ache", "achy joints", "arthralgia", "painful joints",
                     "stiff joints", "swollen joints"],
        "urgency_hint": "yellow",
        "category": "musculoskeletal",
        "red_flag": False,
    },
    "anxiety": {
        "icd10": "F41.9",
        "body_location": "systemic",
        "synonyms": ["worried", "anxious", "panic", "nervous", "stress",
                     "panic attack", "feeling anxious", "on edge"],
        "urgency_hint": "yellow",
        "category": "mental_health",
        "red_flag": False,
    },
    "painful urination": {
        "icd10": "R30.9",
        "body_location": "urinary",
        "synonyms": ["burning urination", "burning when peeing", "dysuria",
                     "pain when urinating", "stinging urination"],
        "urgency_hint": "yellow",
        "category": "urological",
        "red_flag": False,
    },
    "high blood sugar": {
        "icd10": "R73.09",
        "body_location": "systemic",
        "synonyms": ["hyperglycemia", "elevated blood sugar", "high glucose",
                     "blood sugar too high", "sugar levels high"],
        "urgency_hint": "orange",
        "category": "endocrine",
        "red_flag": False,
    },
}


# ── Elasticsearch setup ───────────────────────────────────────────────────────

def get_es_client() -> Elasticsearch:
    es = Elasticsearch(ES_HOST)
    if not es.ping():
        raise ConnectionError(
            f"Cannot connect to Elasticsearch at {ES_HOST}\n"
            "Start it with: docker run -d --name es-neurohealth -p 9200:9200 "
            "-e 'discovery.type=single-node' "
            "-e 'xpack.security.enabled=false' "
            "docker.elastic.co/elasticsearch/elasticsearch:8.13.0"
        )
    return es


def create_index(es: Elasticsearch):
    """Create index with mapping. Drop and recreate if exists."""
    with open("es_mappings/neurohealth_index.json") as f:
        mapping = json.load(f)

    # Adjust dims for dense_vector
    mapping["mappings"]["properties"]["embedding"]["dims"] = 384

    if es.indices.exists(index=INDEX):
        es.indices.delete(index=INDEX)
        print(f"Dropped existing index: {INDEX}")

    es.indices.create(index=INDEX, body=mapping)
    print(f"Created index: {INDEX}")


# ── Embedding ─────────────────────────────────────────────────────────────────

_model = None
def get_model():
    global _model
    if _model is None:
        print("Loading embedding model...")
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def embed_text(text: str) -> list[float]:
    return get_model().encode(text, normalize_embeddings=True).tolist()


# ── MedlinePlus symptom enrichment ───────────────────────────────────────────

def fetch_medlineplus_summary(symptom: str) -> str:
    """Fetch a short NIH summary for this symptom."""
    url = "https://wsearch.nlm.nih.gov/ws/query"
    params = {"db": "healthTopics", "term": symptom, "retmax": "1"}
    try:
        import xml.etree.ElementTree as ET, re
        r = requests.get(url, params=params, timeout=10)
        root = ET.fromstring(r.text)
        el = root.find(".//content[@name='FullSummary']")
        if el is None:
            el = root.find(".//content[@name='snippet']")
        if el is not None and el.text:
            return re.sub(r"<[^>]+>", " ", el.text).strip()[:600]
    except Exception:
        pass
    return ""


# ── Document builder ──────────────────────────────────────────────────────────

def build_symptom_doc(name: str, meta: dict, summary: str = "") -> dict:
    """Build a single ES document for one symptom."""
    all_text_parts = [name] + meta["synonyms"]
    if summary:
        all_text_parts.append(summary)

    patient_text = (
        f"I have {name}. "
        f"Also known as: {', '.join(meta['synonyms'][:4])}. "
        f"{summary[:200] if summary else ''}"
    )
    full_text = f"{name} {' '.join(meta['synonyms'])} {summary}"

    return {
        "doc_id":          hashlib.md5(name.encode()).hexdigest()[:12],
        "module":          "symptoms",
        "source":          "symptom_catalog",
        "urgency_tier":    meta["urgency_hint"],
        "symptom_names":   name,
        "symptom_synonyms": " ".join(meta["synonyms"]),
        "body_location":   meta["body_location"],
        "icd10_codes":     [meta["icd10"]],
        "condition_category": meta["category"],
        "patient_text":    patient_text,
        "clinical_text":   summary,
        "full_text":       full_text,
        "embedding":       embed_text(patient_text),
        "red_flags":       meta.get("red_flag", False),
        "recommendation":  _urgency_to_recommendation(meta["urgency_hint"]),
        "metadata": {
            "canonical_name": name,
            "synonyms_count": len(meta["synonyms"]),
        },
        "ingested_at": datetime.utcnow().isoformat(),
    }


def _urgency_to_recommendation(tier: str) -> str:
    return {
        "red":    "Seek emergency care immediately. Call 108 / 112 / 999.",
        "orange": "See a doctor today or visit urgent care.",
        "yellow": "Book a GP appointment within 2-3 days.",
        "green":  "Rest and self-care at home. Monitor symptoms.",
    }.get(tier, "Consult a healthcare provider if unsure.")


# ── Bulk indexer ──────────────────────────────────────────────────────────────

def index_all_symptoms(es: Elasticsearch, fetch_summaries: bool = True):
    """Index all symptoms from the catalog into Elasticsearch."""
    print(f"\nIndexing {len(SYMPTOM_CATALOG)} symptoms...")
    actions = []

    for name, meta in tqdm(SYMPTOM_CATALOG.items()):
        summary = ""
        if fetch_summaries:
            summary = fetch_medlineplus_summary(name)
            time.sleep(0.3)   # rate limiting for MedlinePlus

        doc = build_symptom_doc(name, meta, summary)
        actions.append({
            "_index": INDEX,
            "_id":    doc["doc_id"],
            "_source": doc,
        })

    success, errors = helpers.bulk(es, actions, raise_on_error=False)
    print(f"Indexed: {success} | Errors: {len(errors)}")
    if errors:
        for e in errors[:3]:
            print(f"  Error: {e}")


# ── Query interface ───────────────────────────────────────────────────────────

def search_symptoms(
    es: Elasticsearch,
    query: str,
    n: int = 5,
    urgency_filter: Optional[str] = None,
) -> list[dict]:
    """
    Hybrid search: BM25 + semantic vector + fuzzy, all in one query.
    This is the retrieval function used by NeuroHealth's NLU pipeline.
    """
    query_vec = embed_text(query)

    # Build optional urgency filter
    filter_clause = []
    if urgency_filter:
        filter_clause.append({"term": {"urgency_tier": urgency_filter}})

    es_query = {
        "size": n,
        "query": {
            "bool": {
                "should": [
                    # 1. BM25 full-text on patient_text
                    {
                        "match": {
                            "patient_text": {
                                "query": query,
                                "boost": 1.5,
                            }
                        }
                    },
                    # 2. Fuzzy match for misspellings (e.g. "headche" → "headache")
                    {
                        "fuzzy": {
                            "symptom_names": {
                                "value": query,
                                "fuzziness": "AUTO",
                                "boost": 1.2,
                            }
                        }
                    },
                    # 3. Synonym expansion via medical_analyzer
                    {
                        "match": {
                            "symptom_synonyms": {
                                "query": query,
                                "boost": 1.0,
                            }
                        }
                    },
                ],
                "filter": filter_clause,
                "minimum_should_match": 1,
            }
        },
        # 4. KNN semantic vector search (re-ranked with RRF)
        "knn": {
            "field":         "embedding",
            "query_vector":  query_vec,
            "k":             n,
            "num_candidates": n * 10,
            "boost":         1.0,
            "filter":        filter_clause if filter_clause else None,
        },
        # RRF fusion of BM25 + KNN scores
        "rank": {
            "rrf": {
                "window_size": 50,
                "rank_constant": 20,
            }
        },
    }

    # Remove null filter from knn if empty
    if not filter_clause:
        del es_query["knn"]["filter"]

    response = es.search(index=INDEX, body=es_query)
    results = []
    for hit in response["hits"]["hits"]:
        src = hit["_source"]
        results.append({
            "symptom":       src.get("symptom_names"),
            "urgency":       src.get("urgency_tier"),
            "recommendation":src.get("recommendation"),
            "body_location": src.get("body_location"),
            "red_flag":      src.get("red_flags"),
            "icd10":         src.get("icd10_codes"),
            "score":         hit.get("_score"),
            "patient_text":  src.get("patient_text", "")[:200],
        })
    return results


# ── Demo ──────────────────────────────────────────────────────────────────────

def demo(es: Elasticsearch):
    test_queries = [
        ("my chest feels tight and I cant breathe properly", None),
        ("headche really bad suddenly",          None),     # typo test
        ("tummy hurts after eating",             None),     # colloquial
        ("diabtes blood sugar very high",        None),     # typo + medical
        ("feel dizzy and want to vomit",         None),
        ("worst headache of my life",            "red"),    # red tier filter
    ]

    print("\n=== Symptom Search Demo ===")
    for query, tier in test_queries:
        print(f"\nQuery: '{query}' (filter: {tier or 'none'})")
        hits = search_symptoms(es, query, n=3, urgency_filter=tier)
        for h in hits:
            flag = " [RED FLAG]" if h["red_flag"] else ""
            print(f"  [{h['urgency'].upper()}]{flag} {h['symptom']} — {h['recommendation']}")


def run():
    es = get_es_client()
    create_index(es)
    index_all_symptoms(es, fetch_summaries=True)
    es.indices.refresh(index=INDEX)
    demo(es)
    print(f"\nModule 1 complete. Index '{INDEX}' ready.")
    print("Next: run modules/module2_conditions.py")


if __name__ == "__main__":
    run()
