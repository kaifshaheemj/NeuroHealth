# NeuroHealth — Elasticsearch Knowledge Base
## Complete Setup & Run Guide

### 1. Start Elasticsearch (Docker)
```bash
# Single node, no auth, port 9200
docker run -d \
  --name es-neurohealth \
  -p 9200:9200 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  -e "ES_JAVA_OPTS=-Xms512m -Xmx512m" \
  docker.elastic.co/elasticsearch/elasticsearch:8.13.0

# Verify it's running
curl http://localhost:9200
```

### 2. Install Python dependencies
```bash
pip install elasticsearch sentence-transformers spacy requests tqdm datasets
python -m spacy download en_core_web_sm
```

### 3. Run modules in order (one at a time)
```bash
# Module 1: Symptoms (start here — builds vocabulary)
python modules/module1_symptoms.py

# Module 2: Conditions (references symptom names from Module 1)
python modules/module2_conditions.py

# Module 3: Medications (from MedlinePlus drug API + MedGlance data)
# python modules/module3_medications.py   ← coming next

# Module 4: Conversations (from DDXPlus + ChatDoctor + synthetic)
# python modules/module4_conversations.py ← coming next
```

### 4. Test the hybrid retriever
```bash
python pipeline/retriever.py
```

### 5. What each module indexes

| Module     | Index name                     | Source                    | Records (est.) |
|------------|--------------------------------|---------------------------|----------------|
| Symptoms   | neurohealth_symptoms           | Catalog + MedlinePlus     | ~50-100        |
| Conditions | neurohealth_conditions         | MedlinePlus Connect API   | ~50-100        |
| Medications| neurohealth_medications        | MedlinePlus + MedGlance   | ~200+          |
| Convos     | neurohealth_conversations      | DDXPlus + ChatDoctor       | ~10,000+       |

### 6. How retrieval works

When a user types "my chest hurts and i cant breathe":

1. BM25 matches "chest", "breathe" against symptom_names, patient_text
2. Fuzzy match handles "chst" → "chest" (typos)
3. Synonym analyzer expands "cant breathe" → "dyspnea", "shortness of breath"
4. KNN semantic search finds conceptually similar embeddings
5. RRF (Reciprocal Rank Fusion) merges all four scores
6. Red flags (chest pain = cardiovascular risk) bubble to top
7. Result context is injected into LLM system prompt for RAG

### 7. Integrate with NeuroHealth LLM pipeline

```python
from pipeline.retriever import NeuroHealthRetriever

retriever = NeuroHealthRetriever()

# In your LLM system prompt construction:
def build_system_prompt(user_message: str) -> str:
    rag_context = retriever.build_rag_context(user_message, n=5)

    return f"""You are NeuroHealth, an AI health assistant.
Use the knowledge below to guide your response.
Always recommend professional medical care for serious symptoms.

{rag_context}

RULES:
- Never diagnose. Only suggest possibilities.
- If RED FLAG detected: recommend emergency services immediately.
- Ask 1 clarifying question if symptoms are ambiguous.
"""
```

### Why Elasticsearch over ChromaDB

| Feature            | Elasticsearch    | ChromaDB         |
|--------------------|------------------|------------------|
| Keyword search     | BM25 (excellent) | Not supported    |
| Fuzzy / typos      | Built-in         | Not supported    |
| Synonym expansion  | Analyzer plugin  | Not supported    |
| Semantic search    | KNN dense vector | Vector only      |
| Hybrid (all 3)     | RRF fusion       | Not possible     |
| Scalability        | Cluster-ready    | Local only       |
| Medical synonyms   | Custom analyzer  | Manual only      |
| Filter by urgency  | Native filter    | Where clause     |
