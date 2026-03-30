# Retrieval Trace — Query to Response

How a user's raw message becomes a ranked list of health conditions and a natural-language response.

---

## Architecture at a Glance

```
User Input
  │
  ▼
① LLM Extraction (Groq 8B)     ─── raw text → structured symptoms
  │
  ▼
② Search Gate (state machine)   ─── enough symptoms? → search / ask more
  │
  ▼
③ 3 ES Search Strategies        ─── symptom_map + hybrid + keyword lookup
  │
  ▼
④ Merge & Rank                  ─── weighted scores + multi-source bonus
  │
  ▼
⑤ RAG Prompt Injection          ─── top results → system prompt context
  │
  ▼
⑥ LLM Response (Groq 70B)      ─── generates answer grounded in ES data
```

---

## Worked Example

**User says:** `"I've had a bad headache for 3 days and feel nauseous"`

---

### Step 1 — Streamlit Receives Input

**File:** `app.py`

```
User types → "I've had a bad headache for 3 days and feel nauseous"
                ↓
app.py calls → process_user_turn(user_message, state)
```

State at this point:
```
phase              = INITIAL_COMPLAINT
confirmed_symptoms = []
symptom_details    = {}
followup_round     = 0
```

The call enters `conversation_engine.py:55` — the `INITIAL_COMPLAINT` branch.

---

### Step 2 — Symptom Extraction via LLM

**File:** `agent/symptom_extractor.py`
**Model:** Groq `llama-3.1-8b-instant` (fast, JSON mode)

`conversation_engine.py:60` calls `_safe_extract()` → `extract_symptoms()`.

The function builds two prompts:

**System prompt** (hardcoded in `symptom_extractor.py:6-24`):
```
You are a medical symptom extraction system.
Given a patient's message and the current conversation context, extract:

1. "new_symptoms": List of symptom phrases
2. "negated_symptoms": Symptoms the user explicitly denied
3. "symptom_details": Key-value details about known symptoms
4. "body_systems_mentioned": Any body regions referenced

Normalize symptoms to simple medical terms:
- "my head is killing me" → "headache"
- "feel like throwing up" → "nausea"
- "can't sleep" → "insomnia"
...
```

**User prompt** (built dynamically in `symptom_extractor.py:44-47`):
```
Patient message: "I've had a bad headache for 3 days and feel nauseous"

Extract symptoms, negations, details, and body systems from this message.
```

If the state already has known symptoms, they are injected as context:
```
Already known symptoms: headache
Known details: {"headache": {"duration": "3 days"}}
```
This prevents the LLM from re-extracting what's already known.

**LLM call:** `call_groq_json()` in `llm/groq_client.py` — uses `response_format={"type": "json_object"}` to guarantee valid JSON output.

**LLM returns:**
```json
{
  "new_symptoms": ["headache", "nausea"],
  "negated_symptoms": [],
  "symptom_details": {
    "headache": { "duration": "3 days", "severity": "bad" }
  },
  "body_systems_mentioned": ["head", "stomach"]
}
```

---

### Step 3 — Apply Extraction to State

**File:** `agent/conversation_engine.py:110-117`

`_apply_extraction()` mutates the conversation state:

```python
state.add_symptoms(["headache", "nausea"])
# → state.confirmed_symptoms = ["headache", "nausea"]
#   (lowercased, deduped, checked against negated list)

state.add_symptom_detail("headache", "duration", "3 days")
state.add_symptom_detail("headache", "severity", "bad")
# → state.symptom_details = {"headache": {"duration": "3 days", "severity": "bad"}}
```

State after extraction:
```
phase              = INITIAL_COMPLAINT → transitions to GATHERING_DETAILS
confirmed_symptoms = ["headache", "nausea"]
symptom_details    = {"headache": {"duration": "3 days", "severity": "bad"}}
negated_symptoms   = []
```

---

### Step 4 — Search Gate Decision

**File:** `agent/state.py:56-61`

```python
def should_search(self) -> bool:
    has_enough = len(self.confirmed_symptoms) >= MIN_SYMPTOMS_TO_SEARCH  # default: 2
    exhausted  = self.followup_round >= MAX_FOLLOWUP_ROUNDS              # default: 3
    return has_enough or exhausted
```

Evaluation:
```
len(["headache", "nausea"]) >= 2  →  True
```

**Decision: Search now.** Calls `_execute_search_and_present(state)`.

If only 1 symptom had been extracted, the agent would instead call `_generate_followup(state)` — an LLM-generated follow-up question focused on:
- Round 1: duration and severity
- Round 2: associated symptoms
- Round 3: triggers and medical history

After 3 rounds, it searches regardless (even with 1 symptom).

---

### Step 5 — Search Orchestrator

**File:** `agent/search_orchestrator.py:11-37`

`run_search(state)` fires 3 independent ES strategies:

---

#### Strategy 1: Symptom Map Lookup — `search_by_symptoms()`

**ES Index:** `neurohealth_symptom_map`
**Weight:** 3× (most precise)
**File:** `step3_ingest_health_topics.py:414-467`

This index was pre-built in step2 — every condition's summary was scanned for symptom phrases, and the relationship was flipped into a reverse index:

```
"headache"   → [Migraine, Meningitis, Concussion, Flu, Brain Tumor, ...]  (70 conditions)
"nausea"     → [Food Poisoning, Gastritis, Migraine, Pregnancy, ...]      (45 conditions)
"chest pain" → [Heart Attack, Angina, Pneumonia, GERD, ...]               (41 conditions)
```

**Query construction:**

```python
query_text = ", ".join(["headache", "nausea"])  # → "headache, nausea"
vec = embed_batch(["headache, nausea"])[0]       # → 768-dim BioLord vector
```

**ES query sent:**
```json
{
  "size": 10,
  "query": {
    "bool": {
      "should": [
        { "terms": { "symptom.keyword": ["headache", "nausea"], "boost": 3.0 } },
        { "match": { "symptom": { "query": "headache, nausea", "boost": 1.5 } } }
      ],
      "minimum_should_match": 1
    }
  },
  "knn": {
    "field": "embedding",
    "query_vector": [0.023, -0.118, ...],
    "k": 10,
    "num_candidates": 100
  }
}
```

What each clause does:
```
terms exact match (boost 3.0)
  → finds documents where symptom.keyword is exactly "headache" or "nausea"
  → high-precision, no fuzziness

text match (boost 1.5)
  → standard BM25 on the symptom text field
  → catches partial matches, tokenized forms

KNN (cosine similarity)
  → BioLord embedding of "headache, nausea"
  → finds semantically similar symptom phrases like "head pain", "cephalalgia",
    "queasy", "feeling sick" even if the exact word isn't there
```

**ES returns hits (symptom map entries):**
```
hit 1: "headache"    → conditions: [{Migraine, ...}, {Meningitis, ...}, {Concussion, ...}, ...]
hit 2: "nausea"      → conditions: [{Food Poisoning, ...}, {Gastritis, ...}, {Migraine, ...}, ...]
hit 3: "head pain"   → conditions: [{Migraine, ...}, {Tension Headache, ...}]  (KNN match)
hit 4: "vomiting"    → conditions: [{Food Poisoning, ...}, {Migraine, ...}]    (KNN match)
...
```

**Aggregation across hits** (`step3_ingest_health_topics.py:448-467`):

The function iterates through every hit's `conditions` array and counts how many matched symptom entries point to each condition:

```
Migraine:          found in "headache" + "nausea" + "head pain" → score = 3
Concussion:        found in "headache" + "nausea"               → score = 2
Food Poisoning:    found in "nausea" + "vomiting"               → score = 2
Meningitis:        found in "headache"                          → score = 1
Tension Headache:  found in "head pain"                         → score = 1
```

Returns sorted by score: `[Migraine(3), Concussion(2), Food Poisoning(2), ...]`

---

#### Strategy 2: Hybrid BM25 + KNN Search — `search_health_topics()`

**ES Index:** `neurohealth_health_topics`
**Weight:** 2× (broader, catches free-text)
**File:** `step3_ingest_health_topics.py:470-559`

**Query construction** (`search_orchestrator.py:47-60`):

```python
def _build_natural_query(state):
    # Enriches symptoms with collected details
    parts = []
    for symptom in state.confirmed_symptoms:
        detail = state.symptom_details.get(symptom, {})
        desc = symptom
        if "duration" in detail:  desc += f" for {detail['duration']}"
        if "severity" in detail:  desc += f" ({detail['severity']})"
        if "location" in detail:  desc += f" in {detail['location']}"
        parts.append(desc)
    return ", ".join(parts)

# Result: "headache for 3 days (bad), nausea"
```

**ES query sent:**
```json
{
  "size": 10,
  "query": {
    "bool": {
      "should": [
        {
          "term": {
            "topic_name.keyword": {
              "value": "headache for 3 days (bad), nausea",
              "boost": 5.0
            }
          }
        },
        {
          "multi_match": {
            "query": "headache for 3 days (bad), nausea",
            "fields": [
              "topic_name^3",
              "also_called^2.5",
              "symptoms^2.5",
              "keywords^1.5",
              "summary^1.2",
              "semantic_text",
              "mesh_terms^1.5"
            ],
            "type": "best_fields",
            "fuzziness": "AUTO",
            "boost": 2.0
          }
        },
        {
          "match": {
            "semantic_text": {
              "query": "headache for 3 days (bad), nausea",
              "analyzer": "health_analyzer",
              "boost": 1.5
            }
          }
        },
        {
          "match": {
            "symptoms": {
              "query": "headache for 3 days (bad), nausea",
              "analyzer": "health_analyzer",
              "boost": 2.0
            }
          }
        }
      ],
      "minimum_should_match": 1
    }
  },
  "knn": {
    "field": "embedding",
    "query_vector": [0.041, -0.087, ...],
    "k": 10,
    "num_candidates": 150
  }
}
```

What each clause does:

```
Clause 1 — exact topic name (boost 5.0)
  topic_name.keyword = "headache for 3 days (bad), nausea"
  → won't match anything (no topic has this exact name)
  → exists for when users type exact condition names like "diabetes"

Clause 2 — multi_match across 7 fields (boost 2.0)
  → tokenizes "headache for 3 days (bad), nausea" into terms
  → "headache" matches in the symptoms field of Migraine (symptoms^2.5 boost)
  → "nausea" matches in the symptoms field of Food Poisoning
  → "headache" matches in topic_name of "Headache" topic (topic_name^3 boost)
  → fuzziness: AUTO handles typos like "headche" → "headache"

Clause 3 — health_analyzer on semantic_text (boost 1.5)
  → the custom analyzer expands synonyms at query time:
    "headache" → also matches "cephalalgia, head pain, migraine"
    "nausea"   → also matches "queasiness, feeling sick, upset stomach"
  → a document containing "cephalalgia" now matches even though user said "headache"

Clause 4 — health_analyzer on symptoms field (boost 2.0)
  → same synonym expansion, but specifically on the symptoms array
  → highest signal — directly matches doc's extracted symptom list

KNN — cosine similarity on embedding
  → BioLord embeds "headache for 3 days (bad), nausea" as a medical concept
  → finds topics whose semantic_text embedding is geometrically close
  → catches conceptual matches even when no keywords overlap
  → e.g., finds "Vestibular Migraine" which mentions "dizziness" not "headache"
```

**ES combines BM25 + KNN scores additively** (since we can't use RRF on basic license):
```
Migraine:          BM25=12.3 + KNN=6.1 = 18.4
Tension Headache:  BM25=8.7  + KNN=3.4 = 12.1
Concussion:        BM25=5.9  + KNN=3.9 = 9.8
...
```

Returns: `[{topic_name: "Migraine", score: 18.4, ...}, ...]`

---

#### Strategy 3: Direct Keyword Lookup — `lookup_topic()`

**ES Index:** `neurohealth_health_kwmap`
**Weight:** 1.5×
**File:** `step3_ingest_health_topics.py:400-411`

For each confirmed symptom, checks if it maps to a known topic:

```python
# Symptom: "headache"
es.get(index="neurohealth_health_kwmap", id="headache")
→ { "keyword": "headache", "canonical_name": "Headache" }

# Then fetch the actual topic doc:
search_health_topics("Headache", n=2, topic_type="condition")
→ [{topic_name: "Headache", score: 25.3}, {topic_name: "Tension Headache", score: 11.2}]
```

```python
# Symptom: "nausea"
es.get(index="neurohealth_health_kwmap", id="nausea")
→ { "keyword": "nausea", "canonical_name": "Nausea and Vomiting" }

search_health_topics("Nausea and Vomiting", n=2, topic_type="condition")
→ [{topic_name: "Nausea and Vomiting", score: 28.1}, ...]
```

This is true O(1) — `es.get()` by document ID, no scoring, no ranking. It's a dictionary lookup stored in Elasticsearch.

---

### Step 6 — Merge & Rank

**File:** `agent/search_orchestrator.py:63-118`

All three strategy results are merged into a single scored dict:

```python
scored = {}

# Symptom map results → weight 3x
for r in symptom_results:
    scored[name]["total_score"] += r["score"] * 3.0
    scored[name]["sources"].add("symptom_map")

# Hybrid results → weight 2x
for r in hybrid_results:
    scored[name]["total_score"] += r["score"] * 2.0
    scored[name]["sources"].add("hybrid")

# Direct results → weight 1.5x
for r in direct_results:
    scored[name]["total_score"] += r["score"] * 1.5
    scored[name]["sources"].add("direct")

# Multi-source bonus: +2.0 per additional source
for entry in scored.values():
    entry["total_score"] += (len(entry["sources"]) - 1) * 2.0
```

**Worked scoring:**

```
Condition            symptom_map    hybrid       direct      bonus     TOTAL
                     (×3)           (×2)         (×1.5)      (+2/src)
─────────────────────────────────────────────────────────────────────────────
Tension Headache     1×3 = 3.0     12.1×2=24.2  11.2×1.5    +4.0     47.95
                                                 =16.8       (3 srcs)
Migraine             3×3 = 9.0     18.4×2=36.8  —           +2.0     47.8
                                                              (2 srcs)
Nausea & Vomiting    —             —            28.1×1.5     +0.0     42.15
                                                =42.15       (1 src)
Headache (topic)     —             —            25.3×1.5     +0.0     37.95
                                                =37.95       (1 src)
Concussion           2×3 = 6.0     9.8×2=19.6   —           +2.0     27.6
                                                              (2 srcs)
Food Poisoning       2×3 = 6.0     7.2×2=14.4   —           +2.0     22.4
                                                              (2 srcs)
```

**Key insight:** Tension Headache ranks #1 despite a lower individual score in each strategy — it appeared in **all 3 strategies**, earning +4.0 multi-source bonus. This rewards convergent evidence from independent search paths.

**Final ranked output** (`merged[:5]` via `config.TOP_CONDITIONS_SHOWN`):

```json
[
  {"topic_name": "Tension Headache",    "final_score": 47.95, "search_sources": ["symptom_map", "hybrid", "direct"]},
  {"topic_name": "Migraine",            "final_score": 47.8,  "search_sources": ["symptom_map", "hybrid"]},
  {"topic_name": "Nausea and Vomiting", "final_score": 42.15, "search_sources": ["direct"]},
  {"topic_name": "Headache",            "final_score": 37.95, "search_sources": ["direct"]},
  {"topic_name": "Concussion",          "final_score": 27.6,  "search_sources": ["symptom_map", "hybrid"]}
]
```

---

### Step 7 — RAG Prompt Construction

**File:** `agent/prompts.py`

`build_system_prompt_with_rag(state, results)` constructs the final system prompt:

```python
# build_rag_context_string() formats each result:
for i, r in enumerate(results):
    text += f"{i+1}. {r['topic_name']} ({r['topic_type']})\n"
    text += f"   Body systems: {r['body_systems']}\n"
    text += f"   Symptoms: {r['symptoms']}\n"
    text += f"   Summary: {r['summary'][:300]}\n"
    text += f"   URL: {r['url']}\n"
```

**Full system prompt sent to LLM:**

```
[SYSTEM_PROMPT_BASE — health assistant role, safety rules, emergency
detection, disclaimer requirements]

## Relevant Health Information (from medical database):

1. Tension Headache (condition)
   Body systems: Nervous System
   Symptoms: headache, muscle pain, neck pain, stiffness, stress
   Summary: A tension headache is the most common type of headache.
   It causes mild to moderate pain that feels like a tight band around
   the head. Stress, poor posture, and fatigue are common triggers...
   URL: https://medlineplus.gov/tensionheadache.html

2. Migraine (condition)
   Body systems: Nervous System
   Symptoms: headache, nausea, vomiting, sensitivity to light, aura
   Summary: A migraine is a type of headache with throbbing pain
   usually on one side of the head. It can cause nausea, vomiting,
   and extreme sensitivity to light and sound...
   URL: https://medlineplus.gov/migraine.html

3. Nausea and Vomiting (condition)
   Body systems: Digestive System
   Symptoms: nausea, vomiting, stomach pain, dizziness
   Summary: Nausea is a feeling of sickness in the stomach...
   URL: https://medlineplus.gov/nauseaandvomiting.html

4. Headache (condition)
   ...

5. Concussion (condition)
   ...

Use ONLY the above information to inform your response.
Cite specific conditions. Include MedlinePlus URLs.
Always include medical disclaimer.
```

The LLM never sees the raw database — only the pre-ranked, pre-filtered context window.

---

### Step 8 — LLM Generates Streamed Response

**File:** `agent/conversation_engine.py:153-171`
**Model:** Groq `llama-3.3-70b-versatile`

```python
system = build_system_prompt_with_rag(state, results)
recent_messages = state.messages[-8:]    # last 8 conversation turns

for chunk in call_groq_streaming(system, recent_messages):
    full_response += chunk
    yield chunk                          # → Streamlit renders in real time
```

**LLM generates (grounded in RAG context):**

```
Based on your symptoms — a headache lasting 3 days along with nausea —
here are some conditions that may be relevant:

**Tension Headache** — The most common type of headache, often felt
as a tight band around the head. Stress, poor posture, and fatigue
are common triggers. [More info](https://medlineplus.gov/tensionheadache.html)

**Migraine** — Migraines cause throbbing pain, often on one side,
and can include nausea, vomiting, and sensitivity to light.
[More info](https://medlineplus.gov/migraine.html)

**Concussion** — If you've had any recent head injury, persistent
headache with nausea could indicate a concussion.
[More info](https://medlineplus.gov/concussion.html)

⚠️ *This is for informational purposes only and not a medical
diagnosis. Please consult a healthcare professional.*

Would you like to know more about any of these conditions?
```

The response is stored in `state.messages` and `state.search_results` is cached. If the user asks a follow-up question (e.g., "tell me more about migraine"), the cached results are reused instead of re-querying ES.

---

## Data Flow Diagram — All Files Involved

```
app.py                          ← Streamlit UI, receives user input
  │
  └→ conversation_engine.py     ← state machine, orchestrates the turn
       │
       ├→ symptom_extractor.py  ← sends user text to Groq 8B for extraction
       │    └→ groq_client.py   ← call_groq_json() — JSON mode, fast model
       │
       ├→ state.py              ← ConversationState dataclass, stores symptoms
       │
       ├→ search_orchestrator.py ← fires 3 ES strategies, merges results
       │    └→ es_client.py      ← thin wrapper, lazy-loads step3 functions
       │         └→ step3_ingest_health_topics.py  ← actual ES queries + BioLord embedding
       │              └→ Elasticsearch (3 indices)
       │                   ├─ neurohealth_symptom_map    (pre-computed reverse index)
       │                   ├─ neurohealth_health_topics  (full hybrid BM25+KNN)
       │                   └─ neurohealth_health_kwmap   (alias → canonical O(1) lookup)
       │
       ├→ prompts.py            ← builds RAG-injected system prompt
       │
       └→ groq_client.py        ← call_groq_streaming() — 70B model, streamed
            └→ Streamlit         ← renders chunks in real time
```

---

## What Happens on Follow-Up Turns

### Case A: User asks about results — "Tell me more about migraine"

```
Phase: FOLLOW_UP
  → _safe_extract("Tell me more about migraine")
  → LLM returns: { "new_symptoms": [] }
  → No new symptoms → _generate_followup_answer()
  → Reuses cached state.search_results (no new ES query)
  → Builds RAG prompt with same results
  → LLM generates detailed response about Migraine
```

### Case B: User adds new symptoms — "I also have blurred vision"

```
Phase: FOLLOW_UP
  → _safe_extract("I also have blurred vision")
  → LLM returns: { "new_symptoms": ["blurred vision"] }
  → state.confirmed_symptoms = ["headache", "nausea", "blurred vision"]
  → New symptoms detected → _execute_search_and_present()
  → Runs all 3 ES strategies again with 3 symptoms
  → Results shift: Migraine with Aura ranks higher, Concussion ranks higher
  → New RAG prompt → LLM generates updated response
```

### Case C: User negates a symptom — "No, I don't have a fever"

```
  → LLM returns: { "negated_symptoms": ["fever"] }
  → state.add_negated(["fever"])
  → If "fever" was in confirmed_symptoms, it gets removed
  → Future searches won't include it
```

---

## Summary: Two LLM Calls, Three ES Queries

| Step | What | Model / Index | Purpose |
|------|------|---------------|---------|
| 1 | Extract symptoms | Groq 8B (JSON mode) | Raw text → structured data |
| 2 | Symptom map search | `neurohealth_symptom_map` | Pre-computed symptom → conditions |
| 3 | Hybrid search | `neurohealth_health_topics` | BM25 + KNN + synonyms |
| 4 | Keyword lookup | `neurohealth_health_kwmap` | Alias → canonical O(1) |
| 5 | Merge & rank | In-memory Python | Weighted scoring + multi-source bonus |
| 6 | Generate response | Groq 70B (streaming) | RAG-grounded natural language |

The LLM never searches the database itself — it only sees what Elasticsearch already retrieved and ranked. The LLM's job is (a) understanding the user's raw language and (b) presenting ES results in a conversational way.
