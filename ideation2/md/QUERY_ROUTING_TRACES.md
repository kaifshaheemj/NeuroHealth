# Query Routing Traces — How Every Type of User Query Flows Through NeuroHealth

Every user message follows the same starting path, then branches based on what the user is actually asking.

---

## The Common Entry Point

No matter what the user types, it always starts here:

```
User types something
      |
      v
process_user_turn()                    [conversation_engine.py]
      |
      v
Is it a simple greeting? ("hi", "hello")
      |
  YES → respond instantly, no LLM call
  NO  → call classify_and_extract()    [intent_classifier.py]
              |
              v
        Groq 8B (fast model) returns:
        {
          "intent": "...",             ← what does the user want?
          "entities": [...]            ← what things did they mention?
        }
              |
              v
        Route to the right handler
```

The classifier picks one of 6 intents:

| Intent | Meaning |
|--------|---------|
| `greeting` | Just saying hi |
| `symptom_report` | Describing symptoms they feel |
| `drug_query` | Asking about a medicine |
| `condition_info` | Asking about a disease/condition |
| `general_health` | General wellness question |
| `followup` | Follow-up on previous answer |

---

## Trace 1: Symptom-Based Query

**User says:** `"I've had a bad headache for 3 days and feel nauseous"`

```
Step 1 — Classify
─────────────────
  Groq 8B returns:
  {
    "intent": "symptom_report",
    "entities": ["headache", "nausea"],
    "new_symptoms": ["headache", "nausea"],
    "symptom_details": {"headache": {"duration": "3 days", "severity": "bad"}}
  }

Step 2 — Save to state
──────────────────────
  state.query_intent       = "symptom_report"
  state.confirmed_symptoms = ["headache", "nausea"]
  state.symptom_details    = {"headache": {"duration": "3 days", "severity": "bad"}}

Step 3 — Route
──────────────
  intent is "symptom_report" → _handle_symptom_report()

Step 4 — Symptom pipeline
─────────────────────────
  2 symptoms found (>= MIN_SYMPTOMS_TO_SEARCH) → should_search() = True

Step 5 — Search ES (3 strategies)
─────────────────────────────────
  Strategy 1: Symptom Map (weight 3x)
    search_by_symptoms(["headache", "nausea"])
    → "headache" maps to 70 conditions (Migraine, Meningitis, Concussion, ...)
    → "nausea" maps to 45 conditions (Food Poisoning, Migraine, ...)
    → Aggregate: Migraine matched both → score = 3 × 2 = 6

  Strategy 2: Hybrid BM25 + KNN (weight 2x)
    search_health_topics("headache for 3 days (bad), nausea")
    → BM25 matches "headache" in symptoms fields
    → health_analyzer expands "headache" → "cephalalgia, head pain, migraine"
    → KNN finds semantically similar topics via BioLord embedding
    → Returns: Migraine, Tension Headache, Concussion, ...

  Strategy 3: Keyword Lookup (weight 1.5x)
    lookup_topic("headache") → "Headache"
    lookup_topic("nausea") → "Nausea and Vomiting"
    → Fetches those specific topic pages

  Merge all → rank by weighted score + multi-source bonus

Step 6 — Build RAG prompt
─────────────────────────
  Uses: SYMPTOM RAG template
  Injects top 5 results into LLM system prompt
  Tells LLM: "list conditions associated with these symptoms"

Step 7 — Groq 70B streams response
───────────────────────────────────
  "Based on your symptoms — headache for 3 days with nausea — here are
  conditions commonly associated: Migraine, Tension Headache, ..."
```

**If only 1 symptom was found**, the system asks a follow-up question instead:
```
Round 1: "How long have you had this? How severe is it?"
Round 2: "Do you also experience nausea, light sensitivity, or dizziness?"
Round 3: "Any recent triggers, injuries, or medical history?"
→ After 3 rounds OR 2+ symptoms → search fires
```

---

## Trace 2: Medicine / Drug Query

**User says:** `"Is ibuprofen safe for children?"`

```
Step 1 — Classify
─────────────────
  {
    "intent": "drug_query",
    "entities": ["ibuprofen"]
  }

Step 2 — Save to state
──────────────────────
  state.query_intent       = "drug_query"
  state.mentioned_entities = ["ibuprofen"]

Step 3 — Route
──────────────
  intent is "drug_query" → _handle_info_query(state, "drug_query")
  (NO follow-up questions — searches immediately)

Step 4 — Search ES
──────────────────
  _search_drug(state):

    lookup_topic("ibuprofen")
    → checks neurohealth_health_kwmap for alias
    → found? use canonical name. not found? use "ibuprofen" as-is

    search_health_topics("ibuprofen", topic_type="drug")
    → ES filters to only the 24 drug-category documents
    → BM25 matches "ibuprofen" in summary/keywords (health_analyzer
      expands "ibuprofen" → "ibuprofen, advil, nurofen, motrin")
    → KNN finds Pain Relievers, Over-the-Counter Medicines, Medicines and Children
    → Returns: [Pain Relievers, Medicines and Children, Over-the-Counter Medicines]

    If no drug results found → falls back to broader search without type filter

Step 5 — Build RAG prompt
─────────────────────────
  Uses: DRUG RAG template (not the symptom template)
  Tells LLM to cover:
    - What the medication is used for
    - Side effects
    - Safety info (age restrictions, pregnancy, interactions)
    - General dosage guidance (never exact numbers)
    - When to seek medical attention

Step 6 — Groq 70B streams response
───────────────────────────────────
  "Ibuprofen is a nonsteroidal anti-inflammatory drug (NSAID) commonly
  used for pain relief and reducing fever...

  Regarding children: Ibuprofen can be given to children over 6 months,
  but pediatric dosing differs from adults. Always consult a pediatrician
  for the correct dose based on your child's weight and age..."
```

---

## Trace 3: Disease / Condition Query

**User says:** `"What is diabetes?"`

```
Step 1 — Classify
─────────────────
  {
    "intent": "condition_info",
    "entities": ["diabetes"]
  }

Step 2 — Save to state
──────────────────────
  state.query_intent       = "condition_info"
  state.mentioned_entities = ["diabetes"]

Step 3 — Route
──────────────
  intent is "condition_info" → _handle_info_query(state, "condition_info")

Step 4 — Search ES
──────────────────
  _search_condition(state):

    lookup_topic("diabetes")
    → neurohealth_health_kwmap: "diabetes" → "Diabetes"

    search_health_topics("Diabetes", topic_type="condition")
    → exact topic_name.keyword match (boost 5x) → Diabetes page
    → also matches: Diabetes Type 1, Diabetes Type 2, Gestational Diabetes
    → Returns with full summaries, symptoms, body_systems

    get_related_conditions("Diabetes", n=3)
    → fetches related_topic_names from the Diabetes document
    → returns: Diabetic Eye Problems, Diabetic Kidney Problems, Diabetic Nerve Problems

    Deduplicates and returns top 5

Step 5 — Build RAG prompt
─────────────────────────
  Uses: CONDITION INFO RAG template
  Tells LLM to cover:
    - What the condition is (plain language)
    - Common symptoms
    - Causes and risk factors
    - Treatment approaches (general, not prescriptions)
    - When to see a doctor
    - Related conditions

Step 6 — Groq 70B streams response
───────────────────────────────────
  "Diabetes is a chronic condition that affects how your body processes
  blood sugar (glucose). There are several types:

  **Type 2 Diabetes** — the most common form, where your body doesn't
  use insulin properly...

  **Type 1 Diabetes** — an autoimmune condition where the body doesn't
  produce insulin...

  Common symptoms include: increased thirst, frequent urination,
  fatigue, blurred vision, slow-healing wounds...

  Related conditions to be aware of: diabetic eye problems,
  kidney disease, nerve damage..."
```

---

## Trace 4: Diagnostic / Test Query

**User says:** `"What is an MRI scan used for?"`

```
Step 1 — Classify
─────────────────
  {
    "intent": "general_health",
    "entities": ["mri scan"]
  }
  (Classified as general_health since it's asking about a procedure,
  not a specific condition or drug)

Step 2 — Route
──────────────
  intent is "general_health" → _handle_info_query(state, "general_health")

Step 3 — Search ES
──────────────────
  _search_general(state):

    search_health_topics("mri scan")        ← no topic_type filter
    → searches ALL 1000+ topics
    → BM25 matches "MRI" in topic_name, summary, keywords
    → KNN semantic search finds related imaging/diagnostic topics
    → Returns: MRI Scans, Diagnostic Imaging, X-Rays, CT Scans, ...

Step 4 — Build RAG prompt
─────────────────────────
  Uses: GENERAL HEALTH RAG template
  Tells LLM: "Provide a helpful, evidence-based response"

Step 5 — Response
─────────────────
  "An MRI (Magnetic Resonance Imaging) scan is a diagnostic test that
  uses powerful magnets and radio waves to create detailed images of
  your organs and tissues..."
```

---

## Trace 5: Prevention / Wellness Query

**User says:** `"How can I prevent heart disease?"`

```
Step 1 — Classify
─────────────────
  {
    "intent": "general_health",
    "entities": ["heart disease", "prevention"]
  }

Step 2 — Route
──────────────
  intent is "general_health" → _handle_info_query(state, "general_health")

Step 3 — Search ES
──────────────────
  _search_general(state):

    search_health_topics("heart disease, prevention")
    → no topic_type filter — searches everything
    → BM25 matches across:
        - "Heart Diseases" (topic_name match)
        - "How to Prevent Heart Disease" (summary content)
        - wellness topics about exercise, nutrition, cholesterol
    → health_analyzer expands: "heart disease" → "heart disease,
      myocardial infarction, cardiac arrest"
    → KNN embedding catches conceptual matches: High Blood Pressure,
      Cholesterol, Exercise, Nutrition
    → Returns: Heart Diseases, High Cholesterol, Exercise,
      How to Lower Cholesterol, High Blood Pressure

Step 4 — Build RAG prompt
─────────────────────────
  Uses: GENERAL HEALTH RAG template
  "Provide a helpful, evidence-based response. Be practical and actionable."

Step 5 — Response
─────────────────
  "Here are evidence-based ways to reduce your risk of heart disease:

  1. **Exercise regularly** — aim for 150 minutes of moderate activity per week
  2. **Eat heart-healthy foods** — fruits, vegetables, whole grains, limit saturated fats
  3. **Monitor blood pressure** — high blood pressure is a major risk factor
  4. **Manage cholesterol** — high LDL cholesterol contributes to plaque buildup
  ..."
```

---

## Trace 6: Follow-Up Query

**User says:** (after getting results about diabetes) `"Tell me more about the symptoms"`

```
Step 1 — Classify
─────────────────
  {
    "intent": "followup",
    "entities": []
  }

Step 2 — Save to state
──────────────────────
  state.query_intent stays "condition_info" (NOT overwritten to "followup")
  This is key — followup inherits the previous intent

Step 3 — Route
──────────────
  intent is "followup" → _handle_followup()

Step 4 — Reuse cached results
─────────────────────────────
  state.search_results already has the diabetes search results
  → NO new ES query needed

Step 5 — Build RAG prompt
─────────────────────────
  Uses previous intent's template → CONDITION INFO RAG template
  (because state.query_intent is still "condition_info")

Step 6 — Response
─────────────────
  LLM sees: the same diabetes ES results + the full conversation history
  → generates a deeper dive into diabetes symptoms specifically
```

---

## Trace 7: Mid-Conversation Intent Switch

**User starts with symptoms, then asks about a drug:**

```
Turn 1: "I have a headache and nausea"
  → intent: symptom_report
  → enters symptom pipeline (follow-up questions, eventually search)
  → state.confirmed_symptoms = ["headache", "nausea"]

Turn 2: Agent asks "How long have you had the headache?"

Turn 3: "About 3 days. By the way, is aspirin safe to take for this?"
  → Classifier sees: "aspirin" + "safe to take"
  → intent: drug_query
  → entities: ["aspirin"]

  What happens:
  - state.query_intent switches from "symptom_report" → "drug_query"
  - state.confirmed_symptoms = ["headache", "nausea"] (PRESERVED)
  - _handle_info_query("drug_query") runs
  - Searches drug index for aspirin
  - Responds with aspirin safety info

Turn 4: "OK, back to my headache. I also have light sensitivity"
  → intent: symptom_report
  → new_symptoms: ["light sensitivity"]
  → state.confirmed_symptoms = ["headache", "nausea", "light sensitivity"]
  → symptom pipeline resumes with 3 symptoms → should_search() = True
  → Full condition search runs
```

The intent switch happens automatically. Every turn classifies fresh. Accumulated symptoms are never lost.

---

## Trace 8: Greeting

**User says:** `"Hello"`

```
Step 1 — Fast path check
────────────────────────
  _is_greeting("Hello") → True (matches the hardcoded set)
  → NO LLM call at all
  → Instant static response

Step 2 — Response
─────────────────
  "Hello! I'm NeuroHealth, your health information assistant.
  I can help you understand symptoms, look up conditions,
  find medication information, and answer general health questions.

  How can I help you today?"

  Cost: 0 LLM calls, 0 ES queries
```

---

## Trace 9: Ambiguous Query

**User says:** `"headache medicine"`

```
Step 1 — Classify
─────────────────
  This is ambiguous — is it a symptom report or a drug query?
  The classifier prompt says: "prefer the most specific intent"
  → drug_query > general_health

  {
    "intent": "drug_query",
    "entities": ["headache medicine"]
  }

Step 2 — Search
───────────────
  _search_drug(state):
    lookup_topic("headache medicine") → probably not in kwmap
    search_health_topics("headache medicine", topic_type="drug")
    → matches: Pain Relievers, Over-the-Counter Medicines
    → health_analyzer: "headache" → "headache, cephalalgia, head pain, migraine"
    If no drug results → fallback broader search finds Headache topic too

Step 3 — Response
─────────────────
  Drug RAG template → response about pain relief medications for headaches
```

---

## Trace 10: Disease Interaction Question

**User says:** `"Can I take metformin if I have kidney disease?"`

```
Step 1 — Classify
─────────────────
  {
    "intent": "drug_query",
    "entities": ["metformin", "kidney disease"]
  }
  (drug_query because the core question is about a medication)

Step 2 — Search
───────────────
  _search_drug(state):
    For "metformin":
      lookup_topic("metformin") → "Diabetes Medicines" (or not found)
      search_health_topics("metformin", topic_type="drug")
      → Diabetes Medicines page (mentions metformin in summary)
    For "kidney disease":
      lookup_topic("kidney disease") → "Kidney Diseases"
      search_health_topics("kidney disease", topic_type="drug")
      → probably no drug results for this
      → fallback: search_health_topics("kidney disease") → Kidney Diseases page

Step 3 — RAG Prompt
───────────────────
  Drug template + both medication and condition results in context
  LLM sees: Diabetes Medicines info + Kidney Diseases info + user's question

Step 4 — Response
─────────────────
  "Metformin is commonly prescribed for Type 2 diabetes, but it requires
  special consideration for people with kidney disease. Kidney function
  affects how the body processes metformin...

  Consult your doctor who can evaluate your kidney function and determine
  if metformin is appropriate for you."
```

---

## Summary: What Happens for Each Query Type

```
                                ┌─────────────────────────┐
                                │    User types message    │
                                └────────────┬────────────┘
                                             │
                                  ┌──────────▼──────────┐
                                  │ Is simple greeting?  │
                                  └──┬──────────────┬────┘
                                 YES │              │ NO
                                     ▼              ▼
                            Static response    Groq 8B classifies
                            (0 LLM calls)      intent + entities
                                                    │
                        ┌───────────┬───────────┬───┴───┬───────────┬──────────┐
                        ▼           ▼           ▼       ▼           ▼          ▼
                   greeting    symptom      drug     condition   general    followup
                      │        report       query      info      health       │
                      │           │           │         │           │          │
                      ▼           ▼           ▼         ▼           ▼          ▼
                   Static    Follow-up    Search     Search     Search     Reuse
                   reply     questions    drugs     conditions  all topics cached
                             then          │         │           │        results
                             search        │         │           │          │
                               │           ▼         ▼           ▼          │
                               │      Drug RAG   Condition   General       │
                               │      template   RAG tmpl    RAG tmpl      │
                               │           │         │           │          │
                               ▼           ▼         ▼           ▼          ▼
                           Symptom     Groq 70B  Groq 70B   Groq 70B   Groq 70B
                           RAG tmpl   streams    streams    streams    streams
                               │      response   response   response   response
                               ▼
                           Groq 70B
                           streams
                           response
```

---

## Cost Per Query Type

| Query Type | LLM Calls | ES Queries | Follow-up Questions? |
|------------|-----------|------------|---------------------|
| Greeting | 0 | 0 | No |
| Drug query | 2 (classify + response) | 1-2 | No — answers immediately |
| Condition info | 2 (classify + response) | 2-3 (search + related) | No — answers immediately |
| General health | 2 (classify + response) | 1 | No — answers immediately |
| Symptom report | 2-8 (classify + follow-ups + response) | 3 (symptom map + hybrid + lookup) | Yes — up to 3 rounds |
| Follow-up | 2 (classify + response) | 0 (uses cached) | No |

Drug, condition, and general queries skip the follow-up loop entirely. Only symptom reports go through the multi-turn gathering process.

---

## What Each File Does

```
User types message
     │
     ▼
app.py ─────────────────────── Streamlit UI, receives input
     │
     ▼
conversation_engine.py ──────── Routes: greeting? classify intent → pick handler
     │
     ├──▶ intent_classifier.py  One Groq 8B call → {intent, entities}
     │
     ├──▶ _handle_info_query()  For drug / condition / general
     │    │
     │    ├──▶ search_orchestrator.py
     │    │    ├── _search_drug()       → search_health_topics(topic_type="drug")
     │    │    ├── _search_condition()   → search_health_topics(topic_type="condition")
     │    │    └── _search_general()     → search_health_topics() [no filter]
     │    │         │
     │    │         └──▶ es_client.py → step3_ingest_health_topics.py → Elasticsearch
     │    │
     │    └──▶ prompts.py → picks DRUG / CONDITION / GENERAL RAG template
     │         └──▶ groq_client.py → Groq 70B streams response
     │
     ├──▶ _handle_symptom_report()  For symptoms
     │    │
     │    ├── Not enough symptoms? → _generate_followup() → Groq 70B asks question
     │    └── Enough symptoms? → _execute_search_and_present()
     │         ├──▶ search_orchestrator.py → run_search() → 3 strategies + merge
     │         └──▶ prompts.py → SYMPTOM RAG template → Groq 70B streams response
     │
     └──▶ _handle_followup()  For follow-ups
          └── Uses cached state.search_results + previous intent's RAG template
```
