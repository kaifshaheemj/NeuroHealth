import os
from dotenv import load_dotenv
# Load environment variables from .env file if it exists
load_dotenv()

# ── Elasticsearch ──────────────────────────────────────────────────────────────
ES_HOST = os.getenv("ES_HOST")
ES_USER = os.getenv("ES_USER")
ES_PASS = os.getenv("ES_PASS")
ES_VERIFY_CERTS = os.getenv("ES_VERIFY_CERTS", "False").lower() in ("true", "1", "t")

INDEX_TOPICS = "neurohealth_health_topics"
INDEX_SYMPTOMS = "neurohealth_symptom_map"
INDEX_KWMAP = "neurohealth_health_kwmap"

# ── Groq ───────────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_FAST_MODEL = "llama-3.1-8b-instant"

# ── Embedding ──────────────────────────────────────────────────────────────────
EMBED_MODEL = "FremyCompany/BioLord-2023-C"
EMBED_DIMS = 768

# ── Agent Behavior ─────────────────────────────────────────────────────────────
MAX_FOLLOWUP_ROUNDS = 3
MIN_SYMPTOMS_TO_SEARCH = 2
MAX_SEARCH_RESULTS = 10
TOP_CONDITIONS_SHOWN = 5
