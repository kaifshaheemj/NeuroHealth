# NeuroHealth

NeuroHealth is a conversational health information assistant.

It helps users:

- Describe symptoms in natural language.
- Refine details through guided follow-up questions.
- Retrieve relevant health topics from MedlinePlus data.
- Get a structured response with safety disclaimers.

This project uses Streamlit for UI, Groq models for language understanding/response generation, and Elasticsearch for hybrid retrieval.


## 1. End-to-End Flow (Simple View)

1. User sends a message in the Streamlit chat UI.
1. The agent classifies intent (symptom report, drug query, condition info, general health, greeting, or follow-up).
1. For symptom reports, the agent extracts structured symptoms and details.
1. The state machine decides whether to ask follow-up questions or run search.
1. Search orchestrator queries Elasticsearch using three strategies.
1. Results are merged and ranked.
1. LLM generates the final response using retrieved context (RAG).

## 2. High-Level Architecture

```text
Streamlit UI (app.py)
  -> Conversation Engine (agent/conversation_engine.py)
     -> Intent Classifier (agent/intent_classifier.py)
     -> Symptom Extractor (agent/symptom_extractor.py)
     -> Search Orchestrator (agent/search_orchestrator.py)
        -> ES Client Wrappers (core/es_client.py)
           -> Preparation/ideation2/step3_ingest_health_topics.py query functions
     -> Prompt Builders (agent/prompts.py)
     -> Groq Client (llm/groq_client.py)
```

## 3. Project Structure

```text
NeuroHealth/
  app.py
  config.py
  requirements.txt
  README.md
  README_2.md

  agent/
    conversation_engine.py
    intent_classifier.py
    symptom_extractor.py
    search_orchestrator.py
    state.py
    prompts.py

  core/
    es_client.py

  llm/
    groq_client.py

  Preparation/
    ideation2/
      step1_retrieve_health_topics.py
      step2_transform_health_topics.py
      step3_ingest_health_topics.py
      output/
```

## 4. Prerequisites

- Python 3.10+
- Elasticsearch 8.x (local or remote)
- Groq API key

## 5. Setup

### 5.1 Install dependencies

```bash
pip install -r requirements.txt
```

### 5.2 Configure environment variables

Create or update `.env` in project root:

```env
ES_HOST=https://127.0.0.1:9200
ES_USER=elastic
ES_PASS=your_es_password
ES_VERIFY_CERTS=False

GROQ_API_KEY=your_groq_api_key
```

Notes:

- `config.py` loads these values via `python-dotenv`.
- If `GROQ_API_KEY` is missing, `app.py` stops with a clear error.

## 6. Build Search Indexes (First-Time Only)

Run from `Preparation/ideation2` in this order:

```bash
cd Preparation/ideation2
python step1_retrieve_health_topics.py
python step2_transform_health_topics.py
python step3_ingest_health_topics.py
```

What each step does:

- `step1_retrieve_health_topics.py`: Downloads MedlinePlus XML datasets and writes raw JSON files into `output/raw`.
- `step2_transform_health_topics.py`: Transforms raw topics into unified semantic schema, extracts symptom signals, and builds `health_topics_semantic.json`, `symptom_condition_map.json`, and `health_keyword_map.json`.
- `step3_ingest_health_topics.py`: Creates Elasticsearch indices, embeds semantic text with `FremyCompany/BioLord-2023-C`, and ingests all records.

Expected Elasticsearch indices:

- `neurohealth_health_topics`
- `neurohealth_symptom_map`
- `neurohealth_health_kwmap`

## 7. Run the App

From project root:

```bash
streamlit run app.py
```

Open:

- `http://localhost:8501`

## 8. Runtime Behavior by Intent

### Symptom report

- Classification + symptom extraction update `ConversationState`.
- Agent asks follow-up questions until either enough symptoms are collected or max rounds are reached.
- Then it runs retrieval and presents likely associated conditions.

### Drug query

- Uses mentioned entities to search drug topics first.
- Falls back to broader search if needed.

### Condition info

- Searches condition topics and related conditions.

### General health

- Runs broad semantic + keyword search on health topics.

### Follow-up

- Uses cached search results when available for conversational continuity.

## 9. Search and Ranking Logic

For symptom-driven flow, three retrieval strategies are combined:

1. Symptom map search (high precision)
1. Natural-language hybrid search (broad recall)
1. Direct alias lookup per symptom

Merge rules:

- Each source contributes weighted score.
- Duplicate conditions are merged.
- Multi-source matches get bonus.
- Final list is sorted by combined score.

## 10. Safety and Medical Boundaries

Prompt and UI rules enforce:

- Informational support only.
- No formal diagnosis.
- No specific medication dosage recommendations.
- Emergency escalation language for red-flag symptoms.
- Mandatory medical disclaimer in responses.

## 11. Quick Troubleshooting

### Elasticsearch connection error

- Verify `ES_HOST`, `ES_USER`, `ES_PASS` in `.env`.
- Confirm the cluster is reachable.

### Empty or poor search results

- Re-run step2 and step3 ingestion pipeline.
- Confirm all three indices exist and have documents.

### Groq errors

- Check `GROQ_API_KEY`.
- Validate network access and quota.

### Slow first search

- Expected if model/indexes are cold.
- Subsequent queries are typically faster.

## 12. Implementation Map (File-by-File)

- `app.py`: Streamlit UI and session state handling.
- `agent/conversation_engine.py`: Central routing and turn processing.
- `agent/intent_classifier.py`: Intent classification and entity extraction.
- `agent/symptom_extractor.py`: JSON-mode symptom extraction.
- `agent/state.py`: State machine and search trigger logic.
- `agent/search_orchestrator.py`: Retrieval strategy orchestration and ranking.
- `core/es_client.py`: Elasticsearch singleton and query-function bridge.
- `llm/groq_client.py`: Groq chat/stream/json wrappers.
- `agent/prompts.py`: Response behavior and RAG templates.
- `Preparation/ideation2/*.py`: ETL, index creation, retrieval primitives.

## 13. Recommended Next Improvements

1. Move hardcoded preparation-script values to environment/config.
1. Add a one-command bootstrap script for local setup.
1. Add automated checks for index health and data freshness.
1. Add tests for intent routing and retrieval merge behavior.

