# NeuroHealth — Health Knowledge Ingestion Pipeline
## MedlinePlus Bulk XML → Elasticsearch (Hybrid Search)

---

### What This Does

Downloads ALL health topics from MedlinePlus (conditions, symptoms, drugs,
wellness, diagnostics — 1000+ topics), transforms them into a unified schema
with extracted symptoms and body-system mappings, and ingests into Elasticsearch
for hybrid BM25 + semantic vector search.

**Core use case:** A user says *"I'm feeling bad because of headache"* →
the conversational agent searches the symptom map → finds Migraine, Tension
Headache, Sinusitis, etc. → asks follow-up questions to narrow down.

---

### Data Sources

| Source | URL | Size | Contents |
|--------|-----|------|----------|
| Health Topics XML | `medlineplus.gov/xml/mplus_topics_YYYY-MM-DD.xml` | ~29 MB | All 1000+ health topics |
| Topic Groups XML | `medlineplus.gov/xml/mplus_topic_groups_YYYY-MM-DD.xml` | ~11 KB | Category/body-system groups |
| Definitions XMLs | `medlineplus.gov/xml/*definitions.xml` | ~44 KB | Fitness, nutrition, vitamins, minerals, general health |

---

### Run Order

```bash
# 0. Install dependencies
pip install requests elasticsearch sentence-transformers tqdm

# 1. Start Elasticsearch (local) OR configure cloud credentials in step3
docker run -d --name es-neurohealth -p 9200:9200 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  docker.elastic.co/elasticsearch/elasticsearch:8.13.0

# 2. Download ALL health topics from MedlinePlus (bulk XML)
python step1_retrieve_health_topics.py
# → output/raw/health_topics_raw.json      (~1000+ topics)
# → output/raw/topic_groups.json           (category groups)
# → output/raw/definitions_raw.json        (health definitions)

# 3. Transform into unified schema with symptom extraction
python step2_transform_health_topics.py
# → output/processed/health_topics_semantic.json   (unified docs)
# → output/processed/symptom_condition_map.json    (symptom → conditions)
# → output/processed/health_keyword_map.json       (alias → canonical name)

# 4. Embed + ingest into Elasticsearch
python step3_ingest_health_topics.py
# → ES index: neurohealth_health_topics    (main hybrid search index)
# → ES index: neurohealth_symptom_map      (symptom → condition lookup)
# → ES index: neurohealth_health_kwmap     (O(1) alias resolution)
```

---

### Unified Document Schema

Every health topic (condition, drug, symptom, wellness, diagnostic, definition)
maps to the same schema:

| Field | Type | Example |
|-------|------|---------|
| `doc_id` | keyword | `nh_cond_b7e2a19f4c` |
| `topic_name` | text + keyword + completion | `"Migraine"` |
| `topic_type` | keyword | `condition` / `drug` / `symptom` / `wellness` / `diagnostic` / `definition` |
| `also_called` | text + keyword | `["Sick Headache"]` |
| `body_systems` | keyword | `["Nervous System"]` |
| `symptoms` | text + keyword | `["headache", "nausea", "blurred vision"]` |
| `mesh_terms` | keyword | `["Migraine Disorders"]` |
| `group_names` | keyword | `["Brain and Nerves"]` |
| `related_topic_names` | text + keyword | `["Headache", "Tension Headache"]` |
| `keywords` | text + keyword | `["migraine", "sick headache", ...]` |
| `summary` | text | First 1500 chars of topic summary |
| `semantic_text` | text | Combined prose optimized for embedding |
| `embedding` | dense_vector (768-dim) | cosine similarity, HNSW indexed (BioLord-2023-C) |
| `url` | keyword | MedlinePlus source URL |

---

### Three Elasticsearch Indices

#### `neurohealth_health_topics` — Main hybrid search index

Supports 5 search modes:
1. **Exact** — `topic_name.keyword` term query (boost 5.0)
2. **BM25** — full-text across name, symptoms, keywords, summary (with `health_analyzer`)
3. **Fuzzy** — handles typos: "pneumona" → "Pneumonia"
4. **Semantic KNN** — embedding search: "bad headache and feel sick" → Migraine
5. **Filtered** — by `topic_type`, `body_systems`, `group_names`

#### `neurohealth_symptom_map` — Symptom → condition lookup

Each entry: one symptom phrase mapped to all conditions that exhibit it.
Also has a 768-dim embedding for semantic symptom matching.

```
"headache" → [Migraine, Tension Headache, Meningitis, Concussion, ...]
"fever"    → [Flu, Meningitis, Malaria, Pneumonia, ...]
```

#### `neurohealth_health_kwmap` — O(1) alias resolution

`GET neurohealth_health_kwmap/_doc/heart%20attack` → `"Heart Attack"`

Covers: synonyms, alternate names, MeSH terms, see-references.

---

### Health Analyzer (Custom ES Analyzer)

Synonym expansion for both symptoms and conditions:

**Symptom synonyms:**
- "headache" ↔ "cephalalgia" ↔ "head pain" ↔ "migraine"
- "fever" ↔ "pyrexia" ↔ "high temperature" ↔ "febrile"
- "fatigue" ↔ "tiredness" ↔ "exhaustion" ↔ "lethargy"

**Condition synonyms:**
- "heart attack" ↔ "myocardial infarction" ↔ "cardiac arrest"
- "flu" ↔ "influenza" ↔ "grippe"
- "high blood pressure" ↔ "hypertension" ↔ "elevated bp"

**Drug synonyms:**
- "paracetamol" ↔ "acetaminophen" ↔ "tylenol"
- "ibuprofen" ↔ "advil" ↔ "motrin"

---

### Integration with NeuroHealth Conversational Agent

```python
from step3_ingest_health_topics import (
    lookup_topic, search_by_symptoms, search_health_topics,
    get_related_conditions, autocomplete_topic
)

# 1. User says: "I'm feeling bad because of headache"
#    Agent extracts symptoms: ["headache"]
#    Agent asks: "Do you also have nausea, fever, or sensitivity to light?"
#    User says: "yes nausea and light sensitivity"
#    Extracted symptoms: ["headache", "nausea", "sensitivity to light"]

# 2. Search symptom map for matching conditions
results = search_by_symptoms(es, ["headache", "nausea", "sensitivity to light"])
# → [{condition: "Migraine", matched_symptoms: ["headache", "nausea"], ...},
#    {condition: "Concussion", ...}, ...]

# 3. Also search health topics with natural language
results = search_health_topics(es, "headache with nausea and light sensitivity")
# → [{topic_name: "Migraine", symptoms: ["headache", "nausea", ...], ...}]

# 4. Get related conditions for deeper exploration
related = get_related_conditions(es, "Migraine")
# → [Headache, Tension Headache, ...]

# 5. Inject into LLM system prompt for RAG response
```

---

### Legacy Drug-Only Pipeline

The original drug-only pipeline files are retained:
- `step1_retrieve_drugs.py` — drug retrieval via search API
- `step2_transform.py` — drug-specific transformation
- `step3_ingest_es.py` — drug-specific ES ingestion

The new pipeline (`*_health_topics.py`) supersedes these for the
symptom→condition conversational agent use case.
