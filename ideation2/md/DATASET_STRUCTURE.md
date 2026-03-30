# Dataset Structure — How Health Data is Organized in Elasticsearch

A visual guide to the 3 Elasticsearch indices, what's stored in each, how they connect, and how queries hit them.

---

## The Big Picture

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                          ELASTICSEARCH CLUSTER                                    │
│                          https://127.0.0.1:9200                                  │
│                                                                                   │
│  ┌─────────────────────────┐ ┌──────────────────────┐ ┌───────────────────────┐  │
│  │                         │ │                      │ │                       │  │
│  │  neurohealth_health     │ │  neurohealth_symptom │ │  neurohealth_health   │  │
│  │  _topics                │ │  _map                │ │  _kwmap               │  │
│  │                         │ │                      │ │                       │  │
│  │  1,114 documents        │ │  1,467 entries       │ │  3,611 entries        │  │
│  │                         │ │                      │ │                       │  │
│  │  The full knowledge     │ │  Symptom → Condition │ │  Alias → Canonical    │  │
│  │  base. Every health     │ │  reverse index.      │ │  name dictionary.     │  │
│  │  topic with summary,    │ │  "headache" maps to  │ │  "flu" → "Flu"        │  │
│  │  symptoms, embedding.   │ │  70 conditions.      │ │  "hbp" → "High Blood  │  │
│  │                         │ │                      │ │   Pressure"           │  │
│  │  Search: BM25 + KNN     │ │  Search: exact +     │ │  Lookup: O(1) by ID   │  │
│  │          hybrid          │ │          semantic    │ │                       │  │
│  │                         │ │                      │ │                       │  │
│  └─────────────────────────┘ └──────────────────────┘ └───────────────────────┘  │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Index 1: `neurohealth_health_topics` — The Main Knowledge Base

**1,114 documents** — every health topic from MedlinePlus.

### What's inside each document

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  DOCUMENT: Migraine                                                         │
│  doc_id: nh_cond_3c3de58f14                                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  IDENTITY                                                                   │
│  ├── topic_name     : "Migraine"                          [text + keyword]  │
│  ├── topic_type     : "condition"                         [keyword filter]  │
│  ├── doc_id         : "nh_cond_3c3de58f14"                [keyword]        │
│  ├── source         : "medlineplus"                       [keyword]        │
│  └── url            : "https://medlineplus.gov/migraine.html"              │
│                                                                             │
│  CLASSIFICATION                                                             │
│  ├── body_systems   : ["Nervous System"]                  [keyword filter]  │
│  ├── group_names    : ["Brain and Nerves"]                [keyword]        │
│  └── mesh_terms     : ["Migraine Disorders"]              [keyword]        │
│                                                                             │
│  ALIASES                                                                    │
│  ├── also_called    : []                                  [text + keyword]  │
│  └── see_references : ["Vascular Headache"]               [text + keyword]  │
│                                                                             │
│  CLINICAL                                                                   │
│  ├── symptoms       : ["anxiety", "cough", "depression",  [text + keyword]  │
│  │                     "increased sensitivity to light",                    │
│  │                     "mood changes", "nausea",                            │
│  │                     "vomiting", "muscle weakness", ...]                  │
│  └── related_topics : ["Headache"]                        [text + keyword]  │
│                                                                             │
│  SEARCH CONTENT                                                             │
│  ├── keywords       : ["migraine", "migraine symptoms",   [text]           │
│  │                     "migraine treatment",                                │
│  │                     "migraine causes",                                   │
│  │                     "migraine disorders",                                │
│  │                     "vascular headache", ...]                            │
│  ├── summary        : "Migraines are a recurring type     [text, 1500ch]   │
│  │                     of headache. They cause moderate                     │
│  │                     to severe pain that is throbbing                     │
│  │                     or pulsing..."                                       │
│  └── semantic_text  : "Migraine is a condition topic.     [text]           │
│                        Body systems: Nervous System.                        │
│                        Symptoms: anxiety, cough,                            │
│                        depression..."                                       │
│                                                                             │
│  EMBEDDING                                                                  │
│  └── embedding      : [0.023, -0.118, 0.041, ...]        [768-dim vector]  │
│                        BioLord-2023-C cosine similarity                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Document count by topic type

```
  875  condition     ████████████████████████████████████████  78.5%
   97  definition    █████                                      8.7%
   50  wellness      ███                                        4.5%
   46  symptom       ██                                         4.1%
   24  drug          █                                          2.2%
   22  diagnostic    █                                          2.0%
  ────
 1,114  total
```

### How fields are used for search

```
User types: "tummy ache"
                │
                ▼
  ┌─ health_analyzer ─────────────────────────────────────────────────┐
  │                                                                    │
  │  "tummy ache"                                                      │
  │       │                                                            │
  │       ▼  synonym expansion                                         │
  │  "tummy ache" + "stomachache" + "abdominal pain" + "belly pain"   │
  │       │                                                            │
  │       ▼  applied to these fields                                   │
  │                                                                    │
  │  topic_name     ──  "Abdominal Pain" matches ✓  (boost x3)       │
  │  also_called    ──  "Bellyache" matches ✓        (boost x2.5)    │
  │  symptoms       ──  matches in many conditions ✓ (boost x2.5)    │
  │  keywords       ──  "stomach ache" matches ✓     (boost x1.5)    │
  │  summary        ──  free-text matches ✓          (boost x1.2)    │
  │  semantic_text  ──  combined prose matches ✓                      │
  │                                                                    │
  └────────────────────────────────────────────────────────────────────┘

  ┌─ KNN vector search (runs in parallel) ────────────────────────┐
  │                                                                │
  │  embed("tummy ache") → [0.034, -0.082, ...]                   │
  │       │                                                        │
  │       ▼  cosine similarity against all 1,114 embeddings        │
  │                                                                │
  │  Finds: Abdominal Pain (0.91), Gastroenteritis (0.84),       │
  │         Appendicitis (0.82), IBS (0.79), ...                  │
  │                                                                │
  └────────────────────────────────────────────────────────────────┘

  ES combines BM25 score + KNN score → final ranked list
```

### The 30 synonym groups in health_analyzer

```
 SYMPTOM SYNONYMS                     CONDITION SYNONYMS
 ─────────────────                    ──────────────────
 headache = cephalalgia               heart attack = myocardial infarction
          = head pain                  stroke = cerebrovascular accident
          = migraine                   diabetes = diabetes mellitus
                                                = blood sugar disease
 stomachache = abdominal pain          high blood pressure = hypertension
            = belly pain               asthma = bronchial asthma
            = tummy ache               flu = influenza = grippe
                                       cold = common cold = URI
 chest pain = angina                   uti = urinary tract infection
           = chest tightness           std = sexually transmitted disease

 fever = pyrexia = high temperature   DRUG SYNONYMS
 fatigue = tiredness = malaise        ────────────────
 nausea = queasiness                  paracetamol = acetaminophen = tylenol
 dizziness = vertigo                  ibuprofen = advil = motrin
 cough = tussis                       aspirin = acetylsalicylic acid
 insomnia = sleeplessness             metformin = glucophage
 anxiety = nervousness
 depression = low mood
 ...
```

When a user types any of these terms, ES automatically expands to match ALL synonyms in that group.

---

## Index 2: `neurohealth_symptom_map` — Symptom-to-Condition Reverse Index

**1,467 entries** — one document per unique symptom phrase.

### What's inside each document

```
┌──────────────────────────────────────────────────────────────────────┐
│  DOCUMENT: headache                                                   │
│  doc_id: sym_9e3779b9                                                │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  symptom         : "headache"                         [text+keyword]  │
│                                                                       │
│  conditions      : [                                  [nested array]  │
│    ┌──────────────────────────────────────────────┐                   │
│    │ condition   : "Anemia"                       │                   │
│    │ body_systems: ["General"]                    │                   │
│    │ topic_type  : "condition"                    │                   │
│    │ url         : ".../anemia.html"              │                   │
│    ├──────────────────────────────────────────────┤                   │
│    │ condition   : "Bird Flu"                     │                   │
│    │ body_systems: ["Immune System",              │                   │
│    │               "Respiratory System"]          │                   │
│    │ topic_type  : "condition"                    │                   │
│    │ url         : ".../birdflu.html"             │                   │
│    ├──────────────────────────────────────────────┤                   │
│    │ condition   : "Brain Tumors"                 │                   │
│    │ body_systems: ["Nervous System"]             │                   │
│    │ ...                                          │                   │
│    ├──────────────────────────────────────────────┤                   │
│    │          ... 70 conditions total ...         │                   │
│    └──────────────────────────────────────────────┘                   │
│                                                                       │
│  condition_count : 70                              [integer]          │
│                                                                       │
│  embedding       : [0.051, -0.093, ...]            [768-dim vector]   │
│                    (embedding of the word "headache")                  │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

### How this index is built (step2)

```
ORIGINAL DATA (in health_topics):

  Migraine
    summary: "...cause nausea and weakness. You may be
              sensitive to light and sound..."
    → extracted symptoms: [nausea, weakness, sensitivity to light]

  Meningitis
    summary: "...symptoms include headache, fever, and
              stiff neck..."
    → extracted symptoms: [headache, fever, stiff neck]

  Flu
    summary: "...symptoms are headache, fever, cough,
              sore throat, body aches..."
    → extracted symptoms: [headache, fever, cough, sore throat]

                    │
                    ▼  FLIP THE RELATIONSHIP

SYMPTOM MAP (reverse index):

  "headache"   → [Migraine, Meningitis, Flu, Brain Tumors, ...]  70 conditions
  "fever"      → [Meningitis, Flu, Malaria, Pneumonia, ...]     112 conditions
  "nausea"     → [Migraine, Food Poisoning, Gastritis, ...]      66 conditions
  "cough"      → [Flu, Common Cold, Bronchitis, Pneumonia, ...]  50 conditions
  "chest pain" → [Heart Attack, Angina, Pneumonia, GERD, ...]    41 conditions
```

### Top symptoms by number of linked conditions

```
  fever .............. 112 conditions  ████████████████████████
  headache ........... 70 conditions   ███████████████
  nausea ............. 66 conditions   ██████████████
  fatigue ............ 58 conditions   ████████████
  cough .............. 50 conditions   ██████████
  diarrhea ........... 48 conditions   ██████████
  chest pain ......... 41 conditions   █████████
  weight loss ........ 39 conditions   ████████
  weakness ........... 37 conditions   ████████
  vomiting ........... 35 conditions   ███████
  rash ............... 34 conditions   ███████
  abdominal pain ..... 33 conditions   ███████
  swelling ........... 31 conditions   ██████
  shortness of breath  29 conditions   ██████
  joint pain ......... 27 conditions   █████
```

### How this index is searched

```
User's symptoms: ["headache", "fever"]

  ES query on neurohealth_symptom_map:
  ┌──────────────────────────────────────────────────────────────┐
  │ BM25:  symptom.keyword IN ["headache", "fever"]  (boost 3x) │
  │ BM25:  symptom MATCH "headache, fever"           (boost 1.5) │
  │ KNN:   cosine(embedding, embed("headache, fever"))           │
  └──────────────────────────────────────────────────────────────┘
         │
         ▼
  Hit 1: "headache" → 70 conditions
  Hit 2: "fever"    → 112 conditions
  Hit 3: "head pain"→ 15 conditions  (KNN semantic match)
         │
         ▼  aggregate across hits

  Meningitis:   found in "headache" + "fever"     → score = 2
  Flu:          found in "headache" + "fever"     → score = 2
  Common Cold:  found in "headache" + "fever"     → score = 2
  Brain Tumors: found in "headache" only          → score = 1
  Malaria:      found in "fever" only             → score = 1
```

---

## Index 3: `neurohealth_health_kwmap` — Alias Dictionary

**3,611 entries** — flat key-value pairs for instant name resolution.

### What's inside each document

```
┌──────────────────────────────────────────────────┐
│  DOCUMENT ID: "flu"                               │
├──────────────────────────────────────────────────┤
│  keyword        : "flu"              [keyword]    │
│  canonical_name : "Flu"              [keyword]    │
│  keyword_text   : "flu"              [text]       │
└──────────────────────────────────────────────────┘
```

That's it — 3 fields. The simplest index. Its power is speed: `es.get(id="flu")` is O(1).

### Real examples from the dataset

```
  ALIAS                          →    CANONICAL NAME
  ─────────────────────────────       ────────────────────────

  CONDITION ALIASES
  "heart attack"                 →    "Heart Attack"
  "myocardial infarction"        →    "Heart Attack"
  "mi"                           →    "Heart Attack"
  "flu"                          →    "Flu"
  "influenza"                    →    "Flu"
  "grippe"                       →    "Flu"
  "diabetes"                     →    "Diabetes"
  "sugar diabetes"               →    "Diabetes"
  "dm"                           →    "Diabetes"
  "high blood pressure"          →    "High Blood Pressure"
  "hypertension"                 →    "High Blood Pressure"
  "hbp"                          →    "High Blood Pressure"
  "htn"                          →    "High Blood Pressure"
  "migraine"                     →    "Migraine"

  SYMPTOM ALIASES
  "bellyache"                    →    "Abdominal Pain"
  "stomach ache"                 →    "Abdominal Pain"
  "pimples"                      →    "Acne"
  "zits"                         →    "Acne"

  MEDICAL TERMS (MeSH)
  "glycated hemoglobin"          →    "A1C"
  "anti-bacterial agents"        →    "Antibiotics"
  "migraine disorders"           →    "Migraine"

  INVERTED FORMS
  "pain, abdominal"              →    "Abdominal Pain"
  "neuroma, acoustic"            →    "Acoustic Neuroma"
  "abortion, induced"            →    "Abortion"
```

### How this index is searched

```
User says: "Tell me about the flu"
                │
                ▼
  lookup_topic("flu")
    → es.get(index="neurohealth_health_kwmap", id="flu")
    → { "keyword": "flu", "canonical_name": "Flu" }
                │
                ▼
  Now the system knows the exact topic name: "Flu"
  → Can fetch the full Flu document from neurohealth_health_topics
```

No scoring. No ranking. Just a dictionary lookup.

---

## How the 3 Indices Connect

```
                        neurohealth_health_kwmap
                        ┌─────────────────────┐
                        │ "flu" → "Flu"        │
User says "flu" ──────▶ │ "influenza" → "Flu"  │──── canonical name
                        │ "grippe" → "Flu"     │         │
                        └─────────────────────┘         │
                                                         ▼
                                               neurohealth_health_topics
                                               ┌───────────────────────┐
                                               │ DOCUMENT: Flu          │
User says "I have a          ┌────────────────▶│ topic_type: condition  │
headache and fever" ─┐       │                 │ symptoms: [headache,   │
                      │       │                 │   fever, cough, ...]   │
                      ▼       │                 │ summary: "Flu is a     │
            neurohealth_symptom_map             │   respiratory..."      │
            ┌───────────────────┐              │ embedding: [...]       │
            │ "headache" → [...,│              └───────────────────────┘
            │   {condition:     │                        ▲
            │    "Flu"}, ...]   │────────────────────────┘
            │ "fever" → [...,   │         conditions point to
            │   {condition:     │         topics in the main index
            │    "Flu"}, ...]   │
            └───────────────────┘
```

The 3 indices form a triangle:
1. **kwmap** resolves any name → canonical topic name in **health_topics**
2. **symptom_map** links symptoms → conditions that exist in **health_topics**
3. **health_topics** is the single source of truth with full content + embeddings

---

## Real Document Examples by Type

### Condition: Migraine

```
┌─ neurohealth_health_topics ────────────────────────────────────────┐
│ topic_name   : "Migraine"                                          │
│ topic_type   : "condition"                                         │
│ body_systems : ["Nervous System"]                                  │
│ also_called  : []                                                  │
│ see_refs     : ["Vascular Headache"]                               │
│ symptoms     : ["anxiety", "depression", "increased sensitivity    │
│                  to light", "mood changes", "nausea",              │
│                  "vomiting", "muscle weakness"]                    │
│ mesh_terms   : ["Migraine Disorders"]                              │
│ related      : ["Headache"]                                        │
│ keywords     : ["migraine", "migraine symptoms",                   │
│                 "migraine treatment", "migraine causes",           │
│                 "vascular headache", "migraine disorders"]         │
│ summary      : "Migraines are a recurring type of headache.       │
│                 They cause moderate to severe pain that is          │
│                 throbbing or pulsing. The pain is often on one     │
│                 side of your head..."                              │
│ embedding    : [768 floats from BioLord-2023-C]                   │
│ url          : https://medlineplus.gov/migraine.html               │
└────────────────────────────────────────────────────────────────────┘
```

### Drug: Antibiotics

```
┌─ neurohealth_health_topics ────────────────────────────────────────┐
│ topic_name   : "Antibiotics"                                       │
│ topic_type   : "drug"                                              │
│ body_systems : ["Immune System"]                                   │
│ symptoms     : ["diarrhea", "nausea", "rash"]   ← side effects    │
│ mesh_terms   : ["Anti-Bacterial Agents"]                           │
│ related      : ["Antibiotic Resistance",                           │
│                 "Bacterial Infections", "Medicines"]               │
│ keywords     : ["antibiotics", "antibiotics medication",           │
│                 "antibiotics side effects"]                        │
│ summary      : "Antibiotics are medicines that fight bacterial     │
│                 infections. They work by killing the bacteria      │
│                 or by making it hard for the bacteria to grow..."  │
│ embedding    : [768 floats]                                        │
│ url          : https://medlineplus.gov/antibiotics.html            │
└────────────────────────────────────────────────────────────────────┘
```

### Wellness: Nutrition

```
┌─ neurohealth_health_topics ────────────────────────────────────────┐
│ topic_name   : "Nutrition"                                         │
│ topic_type   : "wellness"                                          │
│ body_systems : ["General"]                                         │
│ symptoms     : []                            ← no symptoms         │
│ mesh_terms   : ["Nutritional Physiological Phenomena"]             │
│ related      : ["Breastfeeding", "Calcium", "Carbohydrates",      │
│                 "Child Nutrition", "Cholesterol", "Diets",         │
│                 "Obesity", "Vitamins", "Weight Control", ...]      │
│ keywords     : ["nutrition", "healthy eating",                     │
│                 "food and nutrition"]                               │
│ summary      : "Good nutrition is important in keeping people      │
│                 healthy throughout their lives..."                  │
│ embedding    : [768 floats]                                        │
│ url          : https://medlineplus.gov/nutrition.html              │
└────────────────────────────────────────────────────────────────────┘
```

### Diagnostic: A1C

```
┌─ neurohealth_health_topics ────────────────────────────────────────┐
│ topic_name   : "A1C"                                               │
│ topic_type   : "diagnostic"                                        │
│ body_systems : ["Endocrine System"]                                │
│ also_called  : ["Glycohemoglobin", "HbA1C",                       │
│                 "Hemoglobin A1C test"]                              │
│ symptoms     : []                            ← it's a test         │
│ mesh_terms   : ["Glycated Hemoglobin"]                             │
│ keywords     : ["a1c", "hba1c", "hemoglobin a1c",                 │
│                 "glycated hemoglobin"]                              │
│ summary      : "A1C is a blood test for type 2 diabetes           │
│                 and prediabetes. It measures your average           │
│                 blood glucose level over the past 3 months..."     │
│ embedding    : [768 floats]                                        │
│ url          : https://medlineplus.gov/a1c.html                    │
└────────────────────────────────────────────────────────────────────┘
```

---

## Field Types and What They Enable

```
┌─────────────────┬──────────────────┬────────────────────────────────────────┐
│ FIELD TYPE       │ HOW IT'S USED    │ EXAMPLE                               │
├─────────────────┼──────────────────┼────────────────────────────────────────┤
│                  │                  │                                        │
│ text             │ Full-text search │ summary, semantic_text                 │
│ (with analyzer)  │ BM25 scoring     │ User types "throbbing pain" →          │
│                  │ Synonym expand   │ matches "Migraine" summary             │
│                  │                  │                                        │
│ keyword          │ Exact match      │ topic_type = "drug"                    │
│                  │ Filtering        │ body_systems = "Nervous System"        │
│                  │ Aggregations     │ No fuzzy: must match exactly           │
│                  │                  │                                        │
│ text + keyword   │ Both fuzzy and   │ topic_name: "Migraine"                 │
│ (multi-field)    │ exact search     │   .text → fuzzy BM25 match            │
│                  │                  │   .keyword → exact filter/sort         │
│                  │                  │                                        │
│ completion       │ Autocomplete     │ topic_name.suggest                     │
│                  │ Type-ahead       │ User types "dia" → "Diabetes"          │
│                  │                  │                                        │
│ dense_vector     │ KNN semantic     │ embedding: [768 floats]                │
│ (768-dim)        │ Cosine search    │ "tummy ache" finds "Abdominal Pain"   │
│                  │                  │ even without keyword overlap           │
│                  │                  │                                        │
│ nested           │ Complex objects  │ symptom_map.conditions[]               │
│                  │ within array     │ Each condition has its own fields      │
│                  │                  │                                        │
│ integer          │ Numeric filter   │ condition_count: 70                    │
│                  │ Sorting          │ Sort symptoms by # of conditions       │
│                  │                  │                                        │
│ date             │ Timestamp        │ ingested_at: "2026-03-29T..."          │
│                  │                  │                                        │
└─────────────────┴──────────────────┴────────────────────────────────────────┘
```

---

## Which Index Gets Hit by Which Query

```
 USER QUERY                          KWMAP    SYMPTOM_MAP    HEALTH_TOPICS
 ───────────────────────────────     ─────    ───────────    ─────────────

 "Hello"                               -          -              -
  (greeting — no search)

 "I have a headache and nausea"        ✓          ✓              ✓
  (symptom_report — all 3)            1.5x       3.0x           2.0x

 "Is ibuprofen safe for children?"     ✓          -              ✓
  (drug_query)                       lookup               type="drug"

 "What is diabetes?"                   ✓          -              ✓
  (condition_info)                   lookup            type="condition"

 "How to prevent heart disease?"       -          -              ✓
  (general_health)                                       no filter

 "What is an MRI?"                     -          -              ✓
  (general_health)                                       no filter

 "Tell me more about that"             -          -              -
  (followup — uses cached results)
```

---

## Data Pipeline: Source → ES

```
    MedlinePlus XML (29MB)
    https://medlineplus.gov/xml/mplus_topics_2024-03-29.xml
                    │
                    ▼
    ┌──────────────────────────────┐
    │  STEP 1: Retrieve            │
    │  step1_retrieve_health_      │    Downloads 1,017 topics
    │  topics.py                   │    + groups + definitions
    │                              │
    │  Output:                     │
    │  raw/health_topics_raw.json  │
    │  raw/topic_groups.json       │
    │  raw/definitions_raw.json    │
    └──────────────┬───────────────┘
                    │
                    ▼
    ┌──────────────────────────────┐
    │  STEP 2: Transform           │
    │  step2_transform_health_     │    Classifies, extracts symptoms,
    │  topics.py                   │    builds keywords, flips into
    │                              │    symptom map + keyword map
    │  Output:                     │
    │  processed/health_topics_    │    1,114 docs (1,017 + 97 defs)
    │     semantic.json            │
    │  processed/symptom_          │    1,467 symptom entries
    │     condition_map.json       │
    │  processed/health_           │    3,611 keyword entries
    │     keyword_map.json         │
    └──────────────┬───────────────┘
                    │
                    ▼
    ┌──────────────────────────────────────┐
    │  STEP 3: Embed + Ingest              │
    │  step3_ingest_health_topics.py       │
    │                                      │
    │  BioLord-2023-C embeds               │
    │  semantic_text → 768-dim vectors     │
    │                                      │
    │  Creates 3 ES indices                │
    │  Bulk indexes all documents           │
    │  Sets up health_analyzer             │
    │  with 30 synonym groups               │
    │                                      │
    │  ┌────────────────────────────────┐  │
    │  │  neurohealth_health_topics     │  │  1,114 docs + embeddings
    │  │  neurohealth_symptom_map       │  │  1,467 entries + embeddings
    │  │  neurohealth_health_kwmap      │  │  3,611 entries
    │  └────────────────────────────────┘  │
    └──────────────────────────────────────┘
```
