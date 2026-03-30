"""
Module 2: Conditions Extraction & Indexing
============================================
Pulls condition data from MedlinePlus Connect API and indexes
each condition with its associated symptoms, urgency, and care guidelines.

After module1_symptoms.py, run this to populate conditions.
The conditions reference symptom names from Module 1 — that's the link.

Run: python modules/module2_conditions.py
Output: Elasticsearch index "neurohealth_conditions" populated
"""

import json, time, hashlib, requests
from datetime import datetime
from typing import Optional
from tqdm import tqdm
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer

ES_HOST = "http://localhost:9200"
INDEX   = "neurohealth_conditions"
EMBED_MODEL = "all-MiniLM-L6-v2"

# condition name → (ICD10 code, code system, linked symptoms, urgency, category)
CONDITIONS = [
    ("Hypertension",              "I10",    "2.16.840.1.113883.6.90",
     ["headache","dizziness","chest pain","shortness of breath"], "yellow", "cardiovascular"),
    ("Type 2 diabetes",           "E11",    "2.16.840.1.113883.6.90",
     ["fatigue","frequent urination","high blood sugar","blurred vision"], "yellow", "endocrine"),
    ("Asthma",                    "J45.9",  "2.16.840.1.113883.6.90",
     ["shortness of breath","cough","chest tightness","wheezing"], "orange", "respiratory"),
    ("Migraine",                  "G43.9",  "2.16.840.1.113883.6.90",
     ["headache","nausea","sensitivity to light","vomiting"], "yellow", "neurological"),
    ("Gastroenteritis",           "K59.1",  "2.16.840.1.113883.6.90",
     ["nausea","vomiting","diarrhea","abdominal pain","fever"], "yellow", "gastrointestinal"),
    ("Urinary tract infection",   "N39.0",  "2.16.840.1.113883.6.90",
     ["painful urination","frequent urination","lower abdominal pain","fever"], "yellow", "urological"),
    ("Myocardial infarction",     "I21.9",  "2.16.840.1.113883.6.90",
     ["chest pain","shortness of breath","arm pain","nausea","sweating"], "red", "cardiovascular"),
    ("Stroke",                    "I63.9",  "2.16.840.1.113883.6.90",
     ["facial drooping","arm weakness","slurred speech","sudden headache"], "red", "neurological"),
    ("Anaphylaxis",               "T78.2",  "2.16.840.1.113883.6.90",
     ["shortness of breath","rash","throat swelling","dizziness"], "red", "immunological"),
    ("Depression",                "F32.9",  "2.16.840.1.113883.6.90",
     ["fatigue","sadness","insomnia","loss of appetite","anxiety"], "yellow", "mental_health"),
    ("Appendicitis",              "K37",    "2.16.840.1.113883.6.90",
     ["abdominal pain","fever","nausea","vomiting"], "red", "gastrointestinal"),
    ("Lumbar disc herniation",    "M51.1",  "2.16.840.1.113883.6.90",
     ["back pain","leg pain","numbness","weakness"], "yellow", "musculoskeletal"),
    ("Pneumonia",                 "J18.9",  "2.16.840.1.113883.6.90",
     ["fever","cough","shortness of breath","chest pain","fatigue"], "orange", "respiratory"),
    ("Anxiety disorder",          "F41.9",  "2.16.840.1.113883.6.90",
     ["anxiety","palpitations","shortness of breath","chest tightness","dizziness"], "yellow", "mental_health"),
    ("Hypoglycemia",              "E16.0",  "2.16.840.1.113883.6.90",
     ["dizziness","sweating","confusion","shakiness","headache"], "orange", "endocrine"),
]

URGENCY_RECS = {
    "red":    "EMERGENCY: Call 108/112/999 immediately.",
    "orange": "Seek same-day medical attention or urgent care.",
    "yellow": "Book a GP appointment within 2-3 days.",
    "green":  "Monitor at home, self-care measures.",
}

_model = None
def embed(text):
    global _model
    if not _model:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model.encode(text, normalize_embeddings=True).tolist()


def fetch_medlineplus_connect(name, code, codesystem):
    url = "https://connect.medlineplus.gov/service"
    params = {
        "mainSearchCriteria.v.cs": codesystem,
        "mainSearchCriteria.v.c":  code,
        "mainSearchCriteria.v.dn": name,
        "knowledgeResponseType":   "application/json",
        "informationRecipient.languageCode.c": "en",
    }
    try:
        r = requests.get(url, params=params, timeout=12)
        entries = r.json().get("feed", {}).get("entry", [])
        if entries:
            return entries[0].get("summary", {}).get("_value", "")[:800]
    except Exception:
        pass
    return ""


def build_doc(name, code, codesystem, symptoms, urgency, category, summary):
    patient_text = (
        f"I may have {name}. "
        f"Symptoms include: {', '.join(symptoms)}. "
        f"{summary[:250]}"
    )
    return {
        "doc_id":            hashlib.md5(name.encode()).hexdigest()[:12],
        "module":            "conditions",
        "source":            "medlineplus_connect",
        "urgency_tier":      urgency,
        "condition_name":    name,
        "condition_category": category,
        "symptom_names":     ", ".join(symptoms),
        "icd10_codes":       [code],
        "patient_text":      patient_text,
        "clinical_text":     summary,
        "full_text":         f"{name} {' '.join(symptoms)} {summary}",
        "embedding":         embed(patient_text),
        "recommendation":    URGENCY_RECS.get(urgency, ""),
        "red_flags":         urgency == "red",
        "metadata":          {"linked_symptoms": symptoms, "icd10": code},
        "ingested_at":       datetime.utcnow().isoformat(),
    }


def run():
    es = Elasticsearch(ES_HOST)

    # Load mapping and adjust dims
    with open("es_mappings/neurohealth_index.json") as f:
        mapping = json.load(f)
    mapping["mappings"]["properties"]["embedding"]["dims"] = 384

    if es.indices.exists(index=INDEX):
        es.indices.delete(index=INDEX)
    es.indices.create(index=INDEX, body=mapping)
    print(f"Created index: {INDEX}")

    actions = []
    print(f"Indexing {len(CONDITIONS)} conditions...")
    for name, code, codesys, symptoms, urgency, category in tqdm(CONDITIONS):
        summary = fetch_medlineplus_connect(name, code, codesys)
        doc = build_doc(name, code, codesys, symptoms, urgency, category, summary)
        actions.append({"_index": INDEX, "_id": doc["doc_id"], "_source": doc})
        time.sleep(0.4)

    success, errors = helpers.bulk(es, actions)
    es.indices.refresh(index=INDEX)
    print(f"Indexed: {success} | Errors: {len(errors)}")


def search_conditions(es, query, n=5):
    """Hybrid BM25 + vector search for conditions."""
    vec = embed(query)
    body = {
        "size": n,
        "query": {
            "bool": {
                "should": [
                    {"match": {"patient_text": {"query": query, "boost": 1.5}}},
                    {"match": {"symptom_names": {"query": query, "boost": 1.2}}},
                    {"fuzzy": {"condition_name": {"value": query, "fuzziness": "AUTO"}}},
                ]
            }
        },
        "knn": {
            "field": "embedding",
            "query_vector": vec,
            "k": n,
            "num_candidates": n * 10,
        },
        "rank": {"rrf": {"window_size": 50, "rank_constant": 20}},
    }
    hits = es.search(index=INDEX, body=body)["hits"]["hits"]
    return [
        {
            "condition":      h["_source"]["condition_name"],
            "urgency":        h["_source"]["urgency_tier"],
            "symptoms":       h["_source"]["symptom_names"],
            "recommendation": h["_source"]["recommendation"],
            "score":          h["_score"],
        }
        for h in hits
    ]


if __name__ == "__main__":
    run()
    es = Elasticsearch(ES_HOST)
    print("\n=== Condition Search Demo ===")
    for q in ["chest pain shortness of breath", "headache nausea light sensitivity", "diabtes fatigue"]:
        print(f"\nQuery: '{q}'")
        for r in search_conditions(es, q, n=3):
            print(f"  [{r['urgency'].upper()}] {r['condition']} — {r['recommendation']}")
