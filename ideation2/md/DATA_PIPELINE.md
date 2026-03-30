# Data Pipeline: MedlinePlus URL to Elasticsearch

How health data flows from the MedlinePlus website into a searchable Elasticsearch database — step by step, with concrete examples at every stage.

---

## The Big Picture

```
  https://medlineplus.gov/healthtopics.html
                    │
                    ▼
  ┌─────────────────────────────────┐
  │  MedlinePlus Bulk XML Files     │    Source: U.S. National Library of Medicine
  │  (published daily, Tue–Sat)     │    Free, no auth, public domain
  │                                 │
  │  mplus_topics_YYYY-MM-DD.xml    │    ~29 MB, ALL 1000+ health topics
  │  mplus_topic_groups_YYYY-MM-DD  │    ~11 KB, category groupings
  │  *definitions.xml (x5)          │    ~44 KB, fitness/nutrition/vitamins/minerals
  └────────────────┬────────────────┘
                   │
      Step 1       │  Download XML → Parse → JSON
                   ▼
  ┌─────────────────────────────────┐
  │  output/raw/                    │
  │    health_topics_raw.json       │    1000+ topics, every field preserved
  │    topic_groups.json            │    ~50 groups (body systems/categories)
  │    definitions_raw.json         │    ~200 health term definitions
  └────────────────┬────────────────┘
                   │
      Step 2       │  Classify → Extract symptoms → Build keyword maps
                   ▼
  ┌─────────────────────────────────┐
  │  output/processed/              │
  │    health_topics_semantic.json  │    Unified docs with symptoms, body systems, keywords
  │    symptom_condition_map.json   │    "headache" → [Migraine, Meningitis, ...]
  │    health_keyword_map.json      │    "heart attack" → "Heart Attack"
  └────────────────┬────────────────┘
                   │
      Step 3       │  Embed (BioLord-2023-C 768d) → Bulk index
                   ▼
  ┌─────────────────────────────────┐
  │  Elasticsearch                  │
  │    neurohealth_health_topics    │    BM25 + KNN hybrid search
  │    neurohealth_symptom_map      │    Symptom → conditions with embeddings
  │    neurohealth_health_kwmap     │    O(1) alias → canonical name
  └─────────────────────────────────┘
```

---

## Step 1: Retrieve — Download Raw XML

**File:** `step1_retrieve_health_topics.py`

### What it downloads

MedlinePlus publishes daily XML snapshots of their entire health topic database. Instead of paging through a search API, we download the full dump in one request.

| File | URL Pattern | Size | Contents |
|------|-------------|------|----------|
| Health Topics | `medlineplus.gov/xml/mplus_topics_2026-03-19.xml` | ~29 MB | ALL health topics (English) |
| Topic Groups | `medlineplus.gov/xml/mplus_topic_groups_2026-03-19.xml` | ~11 KB | Category groupings |
| Fitness Defs | `medlineplus.gov/xml/fitnessdefinitions.xml` | ~7 KB | Fitness terms |
| General Health Defs | `medlineplus.gov/xml/generalhealthdefinitions.xml` | ~5 KB | General health terms |
| Minerals Defs | `medlineplus.gov/xml/mineralsdefinitions.xml` | ~9 KB | Mineral terms |
| Nutrition Defs | `medlineplus.gov/xml/nutritiondefinitions.xml` | ~14 KB | Nutrition terms |
| Vitamins Defs | `medlineplus.gov/xml/vitaminsdefinitions.xml` | ~9 KB | Vitamin terms |

### How it finds the latest XML

The date in the URL rotates daily (Tue–Sat). The script tries the last 7 days via HEAD requests until it finds a valid URL:

```
https://medlineplus.gov/xml/mplus_topics_2026-03-21.xml  → 404
https://medlineplus.gov/xml/mplus_topics_2026-03-20.xml  → 404 (Sunday)
https://medlineplus.gov/xml/mplus_topics_2026-03-19.xml  → 200 ✓
```

### How it parses the XML

Each `<health-topic>` element in the XML has this structure:

```xml
<health-topic id="4489" language="English" title="Migraine"
              url="https://medlineplus.gov/migraine.html"
              meta-desc="If you suffer from migraine headaches..."
              date-created="04/01/2003">
    <also-called>Sick Headache</also-called>
    <full-summary><![CDATA[<p>If you suffer from migraine headaches...]]></full-summary>
    <group id="15" url="...">Brain and Nerves</group>
    <group id="30" url="...">Head, Eyes, and Ears</group>
    <mesh-heading>
        <descriptor id="D008881">Migraine Disorders</descriptor>
    </mesh-heading>
    <related-topic id="135" url="...">Headache</related-topic>
    <see-reference>Migraine Headache</see-reference>
    <primary-institute url="...">NIH Institute</primary-institute>
    <site title="Migraine" url="...">
        <information-category>Start Here</information-category>
        <organization>Mayo Clinic</organization>
    </site>
</health-topic>
```

The parser extracts every field into a flat JSON record:

```json
{
  "topic_id": "4489",
  "title": "Migraine",
  "url": "https://medlineplus.gov/migraine.html",
  "meta_desc": "If you suffer from migraine headaches...",
  "also_called": ["Sick Headache"],
  "see_references": ["Migraine Headache"],
  "full_summary": "If you suffer from migraine headaches, you are not alone...",
  "full_summary_html": "<p>If you suffer from migraine headaches...",
  "groups": [{"id": "15", "name": "Brain and Nerves", "url": "..."}],
  "group_names": ["Brain and Nerves", "Head, Eyes, and Ears"],
  "mesh_headings": [{"descriptor": "Migraine Disorders", "descriptor_id": "D008881", "qualifiers": []}],
  "mesh_terms": ["Migraine Disorders"],
  "related_topics": [{"id": "135", "name": "Headache", "url": "..."}],
  "related_topic_names": ["Headache"],
  "primary_institute": "NIH Institute",
  "sites": [{"title": "Migraine", "url": "...", "categories": ["Start Here"], "organizations": ["Mayo Clinic"]}],
  "site_count": 15
}
```

Key points:
- HTML tags are stripped from `full_summary` using regex (`<[^>]+>` → space)
- Spanish topics are filtered out (only `language="English"`)
- Definition XMLs use a different schema (`<term-group>` with `<term>` + `<definition>`)

### Output

```
output/raw/
  health_topics_raw.json     ← ~1000+ topic records
  topic_groups.json           ← ~50 group records
  definitions_raw.json        ← ~200 definition records
```

---

## Step 2: Transform — Classify, Extract, Enrich

**File:** `step2_transform_health_topics.py`

Takes the raw JSON from step 1 and produces three enriched files. This is where the intelligence happens — raw medical text becomes structured, searchable knowledge.

### 2a. Topic Type Classification

Each of the 1000+ topics is classified into a type:

```
Raw topic: "Migraine"  →  topic_type: "condition"
Raw topic: "Metformin"  →  topic_type: "drug"
Raw topic: "Fever"      →  topic_type: "symptom"
Raw topic: "MRI Scans"  →  topic_type: "diagnostic"
Raw topic: "Exercise"   →  topic_type: "wellness"
```

**How it works** — two-pass classification:

**Pass 1: Group-based** — Check which MedlinePlus groups the topic belongs to:

| If group contains... | Classify as |
|---|---|
| "Drug Therapy", "Medicines" | `drug` |
| "Symptoms" | `symptom` |
| "Diagnostic Tests", "Laboratory Tests" | `diagnostic` |
| "Wellness and Lifestyle", "Nutrition" | `wellness` |

**Pass 2: Title pattern fallback** — If no group match, check the title with regex:

```python
# Drug patterns: "antibiotic", "vaccine", "inhibitor", "statin", etc.
# Symptom patterns: "pain", "fever", "nausea", "rash", etc.
# Diagnostic patterns: "test", "scan", "biopsy", "x-ray", etc.
```

**Default:** If nothing matches → `condition` (most health topics are conditions/diseases).

### 2b. Body System Mapping

MedlinePlus group names like "Brain and Nerves" are mapped to normalized body system names:

```python
GROUP_TO_BODY_SYSTEM = {
    "brain and nerves":           "Nervous System",
    "heart and blood vessels":    "Cardiovascular System",
    "lungs and breathing":        "Respiratory System",
    "digestive system":           "Digestive System",
    "bones, joints, and muscles": "Musculoskeletal System",
    "skin, hair, and nails":      "Integumentary System",
    "immune system and disorders": "Immune System",
    # ... 30+ mappings total
}
```

A topic can belong to multiple body systems (e.g., "Meningitis" → `["Nervous System", "Immune System"]`). If no mapping matches, it defaults to `["General"]`.

### 2c. Symptom Extraction

This is the most critical part for the conversational agent. For each topic, symptoms are extracted from the summary text using two methods:

**Method 1: Regex pattern matching** — 7 patterns that capture symptom lists from medical writing:

| Pattern | Matches text like... |
|---|---|
| `symptoms include X, Y, and Z` | "Symptoms include headache, nausea, and dizziness" |
| `signs and symptoms include...` | "Signs and symptoms may include fever and chills" |
| `you may experience X` | "You may experience shortness of breath" |
| `it can cause X` | "It can cause fatigue, weight loss, and..." |
| `common symptoms: X` | "Common symptoms are chest pain and..." |

Each match is split on commas/and/or, cleaned, and added to the symptom set.

**Method 2: Known symptom vocabulary scan** — A list of ~90 canonical symptom phrases is checked against the summary:

```python
KNOWN_SYMPTOMS = [
    "headache", "fever", "fatigue", "nausea", "chest pain",
    "shortness of breath", "dizziness", "rash", "joint pain",
    "blurred vision", "insomnia", "anxiety", "depression",
    "seizures", "tremor", "frequent urination", ...
]

# For each symptom: if "headache" in summary.lower() → add it
```

**Deduplication:** Shorter symptoms that are substrings of longer ones are removed (keep "abdominal pain", drop "pain" if both found). Capped at 20 symptoms per topic.

**Example output for "Migraine":**

```json
"symptoms": ["headache", "head pain", "nausea", "vomiting",
             "blurred vision", "dizziness", "numbness", "tingling"]
```

### 2d. Keyword Building

For each topic, an exhaustive keyword set is built for search matching:

```
Input: title="Migraine", also_called=["Sick Headache"],
       mesh=["Migraine Disorders"], symptoms=["headache", "nausea"],
       topic_type="condition"

Output keywords:
  "migraine"                    ← canonical name
  "sick headache"               ← synonym
  "migraine disorders"          ← MeSH term
  "migraine headache"           ← see-reference
  "brain and nerves"            ← group name
  "headache"                    ← symptom
  "nausea"                      ← symptom
  "migraine symptoms"           ← colloquial variation
  "migraine treatment"          ← colloquial variation
  "migraine causes"             ← colloquial variation
```

Drug topics get different variations: `"metformin medicine"`, `"metformin side effects"`.

### 2e. Semantic Text Construction

A single prose string is built per topic, optimized for embedding. This is what the BioLord model vectorizes — it must capture the full "meaning":

```
"Migraine is a condition topic. Also known as: Sick Headache.
Body systems: Nervous System. Symptoms and signs: headache, head pain,
nausea, vomiting, blurred vision, dizziness, numbness, tingling.
Medical terms: Migraine Disorders, Headache. If you suffer from migraine
headaches, you're not alone. About 12 percent of the U.S. population gets them."
```

Structure: `title + type + aliases + body systems + symptoms + MeSH + first 3 sentences of summary + related topics`.

### 2f. Symptom-Condition Map

After processing all topics, a reverse index is built — for every symptom phrase, which conditions mentioned it:

```json
{
  "headache": [
    {"condition": "Migraine", "body_systems": ["Nervous System"], "url": "..."},
    {"condition": "Tension Headache", "body_systems": ["Nervous System"], "url": "..."},
    {"condition": "Meningitis", "body_systems": ["Nervous System", "Immune System"], "url": "..."},
    {"condition": "Concussion", "body_systems": ["Nervous System"], "url": "..."}
  ],
  "fever": [
    {"condition": "Flu", "body_systems": ["Immune System"], "url": "..."},
    {"condition": "Meningitis", ...},
    {"condition": "Pneumonia", ...}
  ],
  "chest pain": [
    {"condition": "Heart Attack", "body_systems": ["Cardiovascular System"], "url": "..."},
    {"condition": "Angina", ...}
  ]
}
```

This is what the conversational agent queries first when a user reports symptoms.

### 2g. Keyword Lookup Map

A flat alias → canonical name dictionary for instant resolution:

```json
{
  "heart attack": "Heart Attack",
  "myocardial infarction": "Heart Attack",
  "flu": "Flu",
  "influenza": "Flu",
  "migraine": "Migraine",
  "sick headache": "Migraine"
}
```

### 2h. Document ID Generation

Each document gets a deterministic ID: `nh_` + first 4 chars of type + `_` + first 10 hex chars of MD5 hash:

```
"Migraine" (condition) → nh_cond_b7e2a19f4c
"Metformin" (drug)     → nh_drug_a3f9e12b1c
```

Deterministic = same topic always gets the same ID, enabling safe re-ingestion.

### Output

```
output/processed/
  health_topics_semantic.json     ← ~1000+ unified documents
  symptom_condition_map.json      ← ~200+ symptom → conditions entries
  health_keyword_map.json         ← ~3000+ alias → canonical name entries
```

---

## Step 3: Ingest — Embed + Index into Elasticsearch

**File:** `step3_ingest_health_topics.py`

### 3a. Embedding with BioLord-2023-C

Each document's `semantic_text` field is converted into a 768-dimensional vector using `FremyCompany/BioLord-2023-C`:

```
"Migraine is a condition topic. Symptoms: headache, nausea..."
    → [0.0234, -0.0891, 0.0412, ..., 0.0156]  (768 floats)
```

**Why this model:**
- Trained on **SNOMED-CT**, **MeSH**, and **ICD-10** medical ontologies
- Understands that "headache" is semantically close to "Migraine" and "cephalalgia"
- "chest pain" maps near "Angina" and "Myocardial Infarction"
- Standard `sentence-transformers` API: `SentenceTransformer("FremyCompany/BioLord-2023-C")`
- 768 dimensions, cosine similarity, ~330MB model download on first run

Embedding is done in batches of 50 documents (32 texts per encode batch internally).

### 3b. Custom Health Analyzer

Before indexing, Elasticsearch is configured with a custom `health_analyzer` that expands medical synonyms at index time AND query time:

```
User searches "headache" → analyzer expands to: headache OR cephalalgia OR head pain OR migraine
User searches "flu"      → analyzer expands to: flu OR influenza OR grippe
User searches "tylenol"  → analyzer expands to: tylenol OR paracetamol OR acetaminophen OR panadol
```

30 synonym groups covering symptoms, conditions, and drugs:

```python
"synonyms": [
    "headache, cephalalgia, head pain, migraine",
    "fever, pyrexia, high temperature, febrile",
    "heart attack, myocardial infarction, cardiac arrest",
    "flu, influenza, grippe",
    "paracetamol, acetaminophen, tylenol, panadol",
    # ... 25 more groups
]
```

### 3c. Three Elasticsearch Indices

**Index 1: `neurohealth_health_topics`** — the main search index

Every health topic as a full document with all fields:

```json
{
  "doc_id": "nh_cond_b7e2a19f4c",
  "topic_name": "Migraine",                              ← text + keyword + autocomplete
  "topic_type": "condition",                              ← keyword (filterable)
  "body_systems": ["Nervous System"],                     ← keyword (filterable)
  "symptoms": ["headache", "nausea", "blurred vision"],   ← text + keyword
  "also_called": ["Sick Headache"],                       ← text + keyword
  "mesh_terms": ["Migraine Disorders"],                   ← keyword
  "keywords": ["migraine", "sick headache", ...],         ← text + keyword
  "summary": "If you suffer from migraine headaches...",  ← text (health_analyzer)
  "semantic_text": "Migraine is a condition topic...",    ← text (health_analyzer)
  "embedding": [0.0234, -0.0891, ...],                   ← dense_vector 768-dim cosine
  "url": "https://medlineplus.gov/migraine.html",
  "ingested_at": "2026-03-21T10:30:00Z"
}
```

Supports 5 search modes:
- **Exact match** — `topic_name.keyword` term query
- **BM25 full-text** — multi-match across all text fields with `health_analyzer`
- **Fuzzy** — `fuzziness: AUTO` handles typos ("pneumona" → "Pneumonia")
- **Semantic KNN** — cosine similarity on the 768-dim embedding vector
- **Filtered** — by `topic_type`, `body_systems`, `group_names`

**Index 2: `neurohealth_symptom_map`** — symptom → conditions

Each symptom phrase is a document with its own embedding:

```json
{
  "symptom": "headache",
  "conditions": [
    {"condition": "Migraine", "body_systems": ["Nervous System"], "url": "..."},
    {"condition": "Tension Headache", ...},
    {"condition": "Meningitis", ...}
  ],
  "condition_count": 12,
  "embedding": [0.0451, -0.0223, ...]
}
```

**Index 3: `neurohealth_health_kwmap`** — O(1) alias resolution

The keyword IS the document ID, so `GET neurohealth_health_kwmap/_doc/heart%20attack` returns immediately:

```json
{
  "keyword": "heart attack",
  "canonical_name": "Heart Attack",
  "keyword_text": "heart attack"
}
```

### 3d. Bulk Indexing

- Health topics: batches of 50 (embed + index per batch)
- Symptom map: batches of 50 (embed + index per batch)
- Keyword map: batches of 1000 (no embedding needed, tiny docs)

Each index is recreated fresh on every run (delete if exists → create → bulk insert → refresh).

---

## Real Scrape Results (Actual Data from MedlinePlus)

> All examples below use **real data** scraped from the live MedlinePlus XML — not mocked.
> Total: **1017 health topics**, **1467 symptom entries**, **3611 keyword aliases**.

### Raw Scrape Output — What Step 1 Produces

Here are 3 real topics exactly as they come out of the XML parser, before any transformation:

**Common Cold** (raw):
```json
{
  "topic_id": "196",
  "title": "Common Cold",
  "url": "https://medlineplus.gov/commoncold.html",
  "meta_desc": "Common cold symptoms usually begin 2 or 3 days after infection and last 2 to 14 days.",
  "date_created": "10/01/1999",
  "also_called": [],
  "see_references": ["Cold, Common"],
  "full_summary": "What is the common cold? The common cold is a mild infection of your upper respiratory tract (which includes your nose and throat). Colds are probably the most common illness. Adults have an average of 2-3 colds per year... What are the symptoms of the common cold? The symptoms of a common cold usually include: Sneezing Stuffy nose (congestion) Runny nose Sore throat Coughing Headache...",
  "groups": [
    {"id": "16", "name": "Infections"},
    {"id": "18", "name": "Ear, Nose and Throat"}
  ],
  "group_names": ["Infections", "Ear, Nose and Throat"],
  "mesh_headings": [{"descriptor": "Common Cold", "descriptor_id": "D003139", "qualifiers": []}],
  "mesh_terms": ["Common Cold"],
  "related_topics": [
    {"id": "107", "name": "Cold and Cough Medicines"},
    {"id": "220", "name": "Flu"},
    {"id": "547", "name": "Sinusitis"},
    {"id": "543", "name": "Viral Infections"}
  ],
  "related_topic_names": ["Cold and Cough Medicines", "Flu", "Sinusitis", "Viral Infections"],
  "primary_institute": "National Institute of Allergy and Infectious Diseases",
  "site_count": 46
}
```

**Heart Attack** (raw):
```json
{
  "topic_id": "5",
  "title": "Heart Attack",
  "url": "https://medlineplus.gov/heartattack.html",
  "date_created": "10/22/1998",
  "also_called": ["MI", "Myocardial infarction"],
  "see_references": ["MI", "Myocardial Infarction"],
  "full_summary": "Each year almost 800,000 Americans have a heart attack. A heart attack happens when blood flow to the heart suddenly becomes blocked... The most common symptoms in men and women are: Chest discomfort. It is often in center or left side of the chest... Shortness of breath... Discomfort in the upper body. You may feel pain or discomfort in one or both arms, the back, shoulders, neck, jaw... You may also have other symptoms, such as nausea, vomiting, dizziness, and lightheadedness...",
  "group_names": ["Blood, Heart and Circulation"],
  "mesh_terms": ["Myocardial Infarction"],
  "related_topic_names": ["Angina", "Cardiac Rehabilitation", "Coronary Artery Bypass Surgery",
                          "Coronary Artery Disease", "Heart Diseases", "Heart Health Tests"]
}
```

**Diabetes** (raw):
```json
{
  "topic_id": "4",
  "title": "Diabetes",
  "url": "https://medlineplus.gov/diabetes.html",
  "date_created": "10/22/1998",
  "also_called": ["DM", "Diabetes mellitus"],
  "see_references": ["Diabetes Mellitus", "Sugar Diabetes"],
  "full_summary": "What is diabetes? Diabetes, also known as diabetes mellitus, is a disease in which your blood glucose, or blood sugar, levels are too high. Glucose is your body's main source of energy... If you have diabetes, your body can't make insulin, can't use insulin as well as it should, or both. Too much glucose stays in your blood... Over time, high blood glucose levels can lead to serious health conditions...",
  "group_names": ["Older Adults", "Endocrine System", "Metabolic Problems", "Diabetes Mellitus"],
  "mesh_terms": ["Diabetes Mellitus"],
  "related_topic_names": ["A1C", "Blood Glucose", "Diabetes Complications", "Diabetes Medicines",
                          "Diabetes Type 1", "Diabetes Type 2", "Diabetic Diet",
                          "Diabetic Eye Problems", "Diabetic Foot", "Diabetic Heart Disease",
                          "Diabetic Kidney Problems", "Diabetic Nerve Problems",
                          "Hyperglycemia", "Hypoglycemia", "Prediabetes"]
}
```

### What's in the raw data (key fields)

| Field | Source | Example |
|-------|--------|---------|
| `title` | XML attribute | `"Heart Attack"` |
| `also_called` | `<also-called>` elements | `["MI", "Myocardial infarction"]` |
| `see_references` | `<see-reference>` elements | `["MI", "Myocardial Infarction"]` |
| `full_summary` | `<full-summary>` text, HTML stripped | Long plain text paragraph |
| `group_names` | `<group>` element text | `["Blood, Heart and Circulation"]` |
| `mesh_terms` | `<mesh-heading>/<descriptor>` text | `["Myocardial Infarction"]` |
| `related_topic_names` | `<related-topic>` element text | `["Angina", "Coronary Artery Disease"]` |
| `sites` | `<site>` elements | External links (Mayo Clinic, NIH, etc.) |

---

## Transformed Output — How Raw Becomes Retrieval-Ready

Step 2 takes each raw record and enriches it with **6 new derived fields** and restructures it for efficient search. Here are the same 3 topics after transformation:

### Common Cold: Raw -> Processed

```
ADDED BY STEP 2:
  topic_type:   "condition"        ← classified from groups ["Infections"]
  body_systems: ["Immune System"]  ← mapped from "Infections"
  symptoms:     ["congestion", "cough", "fever", "headache",
                 "runny nose", "sneezing", "sore throat"]
                                   ← extracted from summary text
  keywords:     ["cold, common", "common cold", "common cold causes",
                 "common cold symptoms", "common cold treatment",
                 "congestion", "cough", "ear, nose and throat",
                 "fever", "headache", "infections",
                 "runny nose", "sneezing", "sore throat"]
                                   ← built from title + synonyms + symptoms
  doc_id:       "nh_cond_7e07d0740a"
                                   ← deterministic MD5-based ID
  semantic_text: "Common Cold is a condition topic. Body systems:
                  Immune System. Symptoms and signs: congestion, cough,
                  fever, headache, runny nose, sneezing, sore throat.
                  Medical terms: Common Cold. What is the common cold?
                  The common cold is a mild infection of your upper
                  respiratory tract..."
                                   ← combined prose optimized for embedding
```

### Heart Attack: Raw -> Processed

```
ADDED BY STEP 2:
  topic_type:   "condition"
  body_systems: ["General"]        ← "Blood, Heart and Circulation" didn't match mapping
  symptoms:     ["dizziness", "lightheadedness", "nausea", "pain",
                 "shortness of breath", "vomiting"]
                                   ← extracted from "symptoms in men and women are..."
  keywords:     ["heart attack", "heart attack causes", "heart attack symptoms",
                 "heart attack treatment", "mi", "mi causes", "mi symptoms",
                 "mi treatment", "myocardial infarction",
                 "myocardial infarction causes", "myocardial infarction symptoms",
                 "myocardial infarction treatment", "nausea", "pain",
                 "shortness of breath", ...]
                                   ← title + also_called + see_references + symptoms
  doc_id:       "nh_cond_778ce81215"
```

### Diabetes: Raw -> Processed

```
ADDED BY STEP 2:
  topic_type:   "condition"
  body_systems: ["Endocrine System", "Geriatrics"]
                                   ← mapped from "Endocrine System" + "Older Adults"
  symptoms:     ["fatigue", "high blood pressure", "numbness", "tingling"]
                                   ← extracted from summary
  keywords:     ["diabetes", "diabetes causes", "diabetes mellitus",
                 "diabetes mellitus causes", "diabetes mellitus symptoms",
                 "diabetes mellitus treatment", "diabetes symptoms",
                 "diabetes treatment", "dm", "dm causes", "dm symptoms",
                 "dm treatment", "sugar diabetes", "fatigue",
                 "high blood pressure", "numbness", "tingling", ...]
                                   ← includes ALL alias permutations
  doc_id:       "nh_cond_c54272db3f"
```

### Side-by-Side: What Changes from Raw to Processed

| Aspect | Raw (Step 1) | Processed (Step 2) | Why |
|--------|-------------|-------------------|-----|
| **Structure** | Flat dump of XML fields | Unified schema with derived fields | Consistent format for all 1017 topics |
| **Classification** | None — just group names | `topic_type: condition/drug/symptom/...` | Enables filtered search by type |
| **Body systems** | Raw group names like "Blood, Heart and Circulation" | Normalized: "Cardiovascular System" | Consistent filtering, cross-topic grouping |
| **Symptoms** | Buried in free-text summary | Extracted list: `["fever", "cough", ...]` | Direct symptom-to-condition matching |
| **Searchability** | Only the raw title and summary text | Keywords, synonyms, colloquial forms | "MI", "heart attack", "myocardial infarction" all resolve |
| **Embedding text** | None | Optimized prose combining all key info | 768-dim vector captures full medical context |
| **Identity** | MedlinePlus topic_id | Deterministic `doc_id` based on content | Safe re-ingestion, deduplication |

---

## Real Symptom-Condition Map (Actual Data)

The symptom map connects symptom phrases to conditions. Here are real entries showing the breadth of coverage:

```
"headache"           -> 70 conditions
  Arteriovenous Malformations (Nervous System)
  Bird Flu (Immune System, Respiratory System)
  Blood Clots (General)
  Common Cold (Immune System)
  Migraine (Nervous System)
  ...

"fever"              -> 112 conditions
  Acute Bronchitis (Immune System, Respiratory System)
  Appendicitis (Digestive System)
  Arthritis (General)
  Bile Duct Cancer (Digestive System, Oncology)
  Ebola (Immune System)
  ...

"chest pain"         -> 41 conditions
  Anaphylaxis (Immune System)
  Angina (Geriatrics)
  Arrhythmia (General)
  Atrial Fibrillation (General)
  Heart Attack (General)
  ...

"nausea"             -> 66 conditions
  Brain Aneurysm (Nervous System)
  Brain Tumors (Nervous System, Oncology)
  Concussion (Nervous System)
  Food Poisoning (Digestive System)
  Migraine (Nervous System)
  ...

"shortness of breath"-> 39 conditions
  Anaphylaxis (Immune System)
  Anxiety (Mental Health)
  Asthma (Immune System, Respiratory System)
  Heart Attack (General)
  Pneumonia (Immune System, Respiratory System)
  ...

"fatigue"            -> 73 conditions
  Addison Disease (Endocrine System, Immune System)
  Anemia (General)
  Diabetes (Endocrine System, Geriatrics)
  Lupus (Immune System)
  Multiple Sclerosis (Nervous System)
  ...

"cough"              -> 61 conditions
"dizziness"          -> 39 conditions
"sore throat"        -> 20 conditions
"rash"               -> 51 conditions
```

**Total: 1467 unique symptom phrases mapped to conditions.**

---

## Real Keyword Alias Map (Actual Data)

The keyword map resolves any name/alias to the canonical topic. Real entries:

```
"heart attack"         -> "Heart Attack"
"mi"                   -> "Heart Attack"
"myocardial infarction"-> "Heart Attack"

"flu"                  -> "Flu"
"influenza"            -> "Flu"

"diabetes"             -> "Diabetes"
"dm"                   -> "Diabetes"
"sugar diabetes"       -> "Diabetes"
"diabetes mellitus"    -> "Diabetes"

"bronchial asthma"     -> "Asthma"

"vascular headache"    -> "Headache"

"cold, common"         -> "Common Cold"
```

**Total: 3611 keyword-to-canonical-name entries.**

---

## End-to-End Trace: URL to Elasticsearch

Tracing a single topic — **Common Cold** — through all 3 steps:

### Source: MedlinePlus XML

```xml
<health-topic id="196" title="Common Cold" url="https://medlineplus.gov/commoncold.html">
    <full-summary>The common cold is a mild infection of your upper respiratory
    tract... The symptoms of a common cold usually include: Sneezing Stuffy
    nose (congestion) Runny nose Sore throat Coughing Headache...</full-summary>
    <group id="16">Infections</group>
    <group id="18">Ear, Nose and Throat</group>
    <mesh-heading><descriptor id="D003139">Common Cold</descriptor></mesh-heading>
    <related-topic id="107">Cold and Cough Medicines</related-topic>
    <related-topic id="220">Flu</related-topic>
    <see-reference>Cold, Common</see-reference>
</health-topic>
```

### Step 1: Parse XML -> Raw JSON

```json
{
  "topic_id": "196",
  "title": "Common Cold",
  "full_summary": "What is the common cold? The common cold is a mild infection...",
  "also_called": [],
  "see_references": ["Cold, Common"],
  "group_names": ["Infections", "Ear, Nose and Throat"],
  "mesh_terms": ["Common Cold"],
  "related_topic_names": ["Cold and Cough Medicines", "Flu", "Sinusitis", "Viral Infections"]
}
```

### Step 2: Classify + Extract + Enrich

```json
{
  "doc_id": "nh_cond_7e07d0740a",
  "topic_name": "Common Cold",
  "topic_type": "condition",
  "body_systems": ["Immune System"],
  "symptoms": ["congestion", "cough", "fever", "headache", "runny nose", "sneezing", "sore throat"],
  "keywords": ["cold, common", "common cold", "common cold symptoms", "congestion", "cough", "fever", "headache", ...],
  "semantic_text": "Common Cold is a condition topic. Body systems: Immune System. Symptoms and signs: congestion, cough, fever, headache, runny nose, sneezing, sore throat. Medical terms: Common Cold. What is the common cold? The common cold is a mild infection of your upper respiratory tract...",
  "url": "https://medlineplus.gov/commoncold.html"
}
```

### Step 3: Embed + Index

```json
{
  ... all fields above, plus:
  "embedding": [0.0234, -0.0891, 0.0412, ..., -0.0156],   // 768 floats from BioLord-2023-C
  "ingested_at": "2026-03-21T10:30:00Z"
}
```

The topic is now sitting in 3 ES indices:
- `neurohealth_health_topics` — full document, searchable by BM25 + KNN
- `neurohealth_symptom_map` — "cough" entry includes Common Cold, "headache" entry includes Common Cold, etc.
- `neurohealth_health_kwmap` — `"cold, common" -> "Common Cold"`, `"common cold" -> "Common Cold"`

### How a User Query Finds It

User says: **"I have a runny nose and sore throat"**

1. **Symptom map search** — queries `neurohealth_symptom_map` for `["runny nose", "sore throat"]`
   - "runny nose" matches: Common Cold, Flu, Allergy, Sinusitis...
   - "sore throat" matches: Common Cold, Flu, Diphtheria, Strep Throat...
   - **Common Cold** appears in BOTH -> highest score

2. **Hybrid BM25+KNN search** — queries `neurohealth_health_topics` with the natural text
   - BM25 matches "runny nose" and "sore throat" in the `symptoms` and `summary` fields
   - KNN finds semantically similar topics via the 768-dim embedding
   - **Common Cold** ranks high on both

3. **Result merged** — Common Cold found by multiple strategies, gets multi-source bonus

---

## Run Commands

```bash
cd Preparation/ideation2

# Step 1: Download (~30 seconds)
python step1_retrieve_health_topics.py

# Step 2: Transform (~5 seconds)
python step2_transform_health_topics.py

# Step 3: Embed + Ingest (~5-10 minutes, embedding is the bottleneck)
python step3_ingest_health_topics.py
```

Dependencies: `pip install requests elasticsearch sentence-transformers tqdm`
