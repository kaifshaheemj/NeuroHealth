"""
Step 2: Transform ALL Health Topics into Unified Semantic Documents
===================================================================
Takes raw data from step1 and produces:

1. Unified semantic JSON  — one doc per health topic (conditions, symptoms,
   drugs, wellness, diagnostics) with extracted symptoms, body systems,
   condition types, and rich semantic text for embedding.

2. Symptom-to-condition map — maps symptom phrases to possible conditions
   so the conversational agent can narrow down user complaints.

3. Keyword lookup map — bidirectional map of colloquial terms, synonyms,
   and alternate names to canonical topic names.

Output:
  output/processed/health_topics_semantic.json
  output/processed/symptom_condition_map.json
  output/processed/health_keyword_map.json

Run: python step2_transform_health_topics.py
"""

import json
import hashlib
import os
import re
from collections import defaultdict
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────

INPUT_TOPICS = "output/raw/health_topics_raw.json"
INPUT_GROUPS = "output/raw/topic_groups.json"
INPUT_DEFS = "output/raw/definitions_raw.json"

OUT_DIR = "output/processed"
OUT_SEMANTIC = os.path.join(OUT_DIR, "health_topics_semantic.json")
OUT_SYMPTOM_MAP = os.path.join(OUT_DIR, "symptom_condition_map.json")
OUT_KEYWORD_MAP = os.path.join(OUT_DIR, "health_keyword_map.json")

os.makedirs(OUT_DIR, exist_ok=True)


# ── Topic Type Classification ────────────────────────────────────────────────

# Group names that indicate specific topic types
DRUG_GROUPS = {
    "drug therapy", "medicines", "complementary and alternative medicine",
    "drug safety", "drug reactions",
}
SYMPTOM_GROUPS = {
    "symptoms",
}
DIAGNOSTIC_GROUPS = {
    "diagnostic tests", "laboratory tests", "screening",
}
WELLNESS_GROUPS = {
    "wellness and lifestyle", "exercise and physical fitness",
    "nutrition", "safety issues",
}

# Title patterns for classification
DRUG_TITLE_PATTERNS = re.compile(
    r"\b(medicine|medication|drug|antibiotic|vaccine|insulin|supplement|"
    r"inhibitor|blocker|agonist|antagonist|statin|steroid|antiviral|"
    r"antibiotic|antifungal|antihistamine|analgesic|anesthetic)\b", re.I
)

SYMPTOM_TITLE_PATTERNS = re.compile(
    r"\b(pain|ache|aching|fever|nausea|vomiting|dizziness|fatigue|"
    r"swelling|bleeding|rash|itching|cough|shortness of breath|"
    r"numbness|tingling|weakness)\b", re.I
)

DIAGNOSTIC_TITLE_PATTERNS = re.compile(
    r"\b(test|scan|screening|x-ray|mri|ct scan|biopsy|blood test|"
    r"ultrasound|endoscopy|colonoscopy|mammogram|diagnosis)\b", re.I
)


def classify_topic_type(title: str, group_names: list[str], summary: str) -> str:
    """Classify a health topic into a type category."""
    lower_groups = {g.lower() for g in group_names}
    lower_title = title.lower()

    # Check group-based classification first
    if lower_groups & DRUG_GROUPS:
        return "drug"
    if lower_groups & SYMPTOM_GROUPS:
        return "symptom"
    if lower_groups & DIAGNOSTIC_GROUPS:
        return "diagnostic"
    if lower_groups & WELLNESS_GROUPS:
        return "wellness"

    # Fallback to title pattern matching
    if DRUG_TITLE_PATTERNS.search(lower_title):
        return "drug"
    if SYMPTOM_TITLE_PATTERNS.search(lower_title):
        return "symptom"
    if DIAGNOSTIC_TITLE_PATTERNS.search(lower_title):
        return "diagnostic"

    # Default: condition (most health topics are conditions/diseases)
    return "condition"


# ── Body System Mapping ──────────────────────────────────────────────────────

GROUP_TO_BODY_SYSTEM = {
    "brain and nerves": "Nervous System",
    "brain disorders": "Nervous System",
    "mental health and behavior": "Mental Health",
    "heart and blood vessels": "Cardiovascular System",
    "heart diseases": "Cardiovascular System",
    "vascular diseases": "Cardiovascular System",
    "blood and blood disorders": "Cardiovascular System",
    "lungs and breathing": "Respiratory System",
    "lung diseases": "Respiratory System",
    "digestive system": "Digestive System",
    "digestive diseases": "Digestive System",
    "stomach and intestinal disorders": "Digestive System",
    "liver diseases": "Digestive System",
    "kidneys and urinary system": "Urinary System",
    "kidney diseases": "Urinary System",
    "bones, joints, and muscles": "Musculoskeletal System",
    "bone diseases": "Musculoskeletal System",
    "joint disorders": "Musculoskeletal System",
    "muscle disorders": "Musculoskeletal System",
    "skin, hair, and nails": "Integumentary System",
    "skin conditions": "Integumentary System",
    "eye diseases": "Eyes",
    "eyes and vision": "Eyes",
    "ear, nose, and throat": "ENT",
    "ear disorders": "ENT",
    "nose disorders": "ENT",
    "throat disorders": "ENT",
    "mouth and teeth": "Oral Health",
    "dental health": "Oral Health",
    "hormones and the endocrine system": "Endocrine System",
    "endocrine diseases": "Endocrine System",
    "thyroid diseases": "Endocrine System",
    "diabetes": "Endocrine System",
    "diabetes mellitus": "Endocrine System",
    "immune system and disorders": "Immune System",
    "autoimmune diseases": "Immune System",
    "infections": "Immune System",
    "infectious diseases": "Immune System",
    "cancers": "Oncology",
    "female reproductive system": "Reproductive System",
    "male reproductive system": "Reproductive System",
    "pregnancy and reproduction": "Reproductive System",
    "sexual health": "Reproductive System",
    "children and teenagers": "Pediatrics",
    "child health": "Pediatrics",
    "infant and newborn care": "Pediatrics",
    "older adults": "Geriatrics",
    "seniors' health": "Geriatrics",
}


def infer_body_systems(group_names: list[str]) -> list[str]:
    """Map group names to body systems."""
    systems = set()
    for gn in group_names:
        lower = gn.lower()
        if lower in GROUP_TO_BODY_SYSTEM:
            systems.add(GROUP_TO_BODY_SYSTEM[lower])
        else:
            # Partial match
            for key, system in GROUP_TO_BODY_SYSTEM.items():
                if key in lower or lower in key:
                    systems.add(system)
                    break
    return sorted(systems) if systems else ["General"]


# ── Symptom Extraction ───────────────────────────────────────────────────────

SYMPTOM_PATTERNS = [
    # "Symptoms include X, Y, and Z"
    re.compile(
        r"symptoms?\s+(?:include|are|may include|can include|of\s+\w+\s+include)\s*:?\s*(.+?)(?:\.\s|$)",
        re.I
    ),
    # "Signs and symptoms include..."
    re.compile(
        r"signs?\s+(?:and|or)\s+symptoms?\s+(?:include|are|may include)\s*:?\s*(.+?)(?:\.\s|$)",
        re.I
    ),
    # "You may experience X"
    re.compile(
        r"you\s+may\s+(?:experience|have|feel|notice|develop)\s+(.+?)(?:\.\s|$)",
        re.I
    ),
    # "It can cause X"
    re.compile(
        r"(?:it|this|the (?:condition|disease|disorder))\s+(?:can|may)\s+cause\s+(.+?)(?:\.\s|$)",
        re.I
    ),
    # "Causes X, Y, and Z"
    re.compile(
        r"causes?\s+(?:symptoms?\s+(?:such as|like|including)\s+)(.+?)(?:\.\s|$)",
        re.I
    ),
    # "Common symptoms: X"
    re.compile(
        r"common\s+symptoms?\s*(?:include|are|:)\s*(.+?)(?:\.\s|$)",
        re.I
    ),
    # "leads to X"
    re.compile(
        r"(?:leads?|leading)\s+to\s+(.+?)(?:\.\s|$)",
        re.I
    ),
]

# Individual symptom phrases to detect in text
KNOWN_SYMPTOMS = [
    "headache", "migraine", "head pain",
    "fever", "chills", "night sweats",
    "fatigue", "tiredness", "weakness", "malaise", "lethargy",
    "nausea", "vomiting", "diarrhea", "constipation",
    "abdominal pain", "stomach pain", "belly pain", "cramps",
    "chest pain", "chest tightness", "palpitations",
    "shortness of breath", "difficulty breathing", "wheezing", "cough",
    "sore throat", "runny nose", "congestion", "sneezing",
    "dizziness", "lightheadedness", "vertigo", "fainting",
    "numbness", "tingling", "pins and needles",
    "back pain", "neck pain", "joint pain", "muscle pain", "stiffness",
    "swelling", "inflammation", "redness", "bruising",
    "rash", "itching", "hives", "blisters", "dry skin",
    "blurred vision", "double vision", "vision loss", "eye pain",
    "hearing loss", "ringing in ears", "tinnitus", "ear pain",
    "difficulty swallowing", "hoarseness", "loss of voice",
    "weight loss", "weight gain", "loss of appetite",
    "insomnia", "difficulty sleeping", "sleepiness", "drowsiness",
    "anxiety", "depression", "mood changes", "irritability", "confusion",
    "memory loss", "difficulty concentrating", "brain fog",
    "seizures", "tremor", "muscle spasms", "twitching",
    "frequent urination", "painful urination", "blood in urine",
    "bleeding", "blood in stool", "bruising easily",
    "hair loss", "brittle nails",
    "high blood pressure", "low blood pressure",
    "high blood sugar", "low blood sugar",
    "swollen lymph nodes", "lump",
    "difficulty walking", "balance problems", "coordination problems",
]


def extract_symptoms(summary: str, title: str) -> list[str]:
    """Extract symptoms mentioned in a health topic summary."""
    if not summary:
        return []

    symptoms = set()
    lower_summary = summary.lower()

    # Pattern-based extraction from sentences
    for pattern in SYMPTOM_PATTERNS:
        matches = pattern.findall(summary)
        for match in matches:
            # Split on commas and 'and'/'or'
            parts = re.split(r",\s*|\s+and\s+|\s+or\s+", match)
            for part in parts:
                cleaned = part.strip().lower()
                cleaned = re.sub(r"[^a-z\s]", "", cleaned).strip()
                if 2 < len(cleaned) < 60:
                    symptoms.add(cleaned)

    # Known symptom matching
    for symptom in KNOWN_SYMPTOMS:
        if symptom in lower_summary:
            symptoms.add(symptom)

    # Deduplicate substrings (keep longer form)
    final = set()
    sorted_symptoms = sorted(symptoms, key=len, reverse=True)
    for s in sorted_symptoms:
        if not any(s != existing and s in existing for existing in final):
            final.add(s)

    return sorted(final)[:20]


# ── Keyword Building ─────────────────────────────────────────────────────────

def build_keywords(title: str, also_called: list[str], see_refs: list[str],
                   mesh_terms: list[str], group_names: list[str],
                   symptoms: list[str], topic_type: str) -> list[str]:
    """Build exhaustive keyword set for a health topic."""
    kws = set()

    # Canonical name
    kws.add(title.lower())

    # All synonyms
    for ac in also_called:
        kws.add(ac.lower())
    for sr in see_refs:
        kws.add(sr.lower())

    # MeSH terms
    for m in mesh_terms:
        kws.add(m.lower())

    # Group names
    for g in group_names:
        kws.add(g.lower())

    # Symptom-based phrases (for conditions)
    if topic_type == "condition":
        for symptom in symptoms[:10]:
            kws.add(symptom)

    # Colloquial variations
    for name in [title.lower()] + [ac.lower() for ac in also_called[:5]]:
        if topic_type == "condition":
            kws.add(f"{name} symptoms")
            kws.add(f"{name} treatment")
            kws.add(f"{name} causes")
        elif topic_type == "drug":
            kws.add(f"{name} medicine")
            kws.add(f"{name} medication")
            kws.add(f"{name} side effects")

    # Remove empty strings
    kws.discard("")

    return sorted(kws)


# ── Semantic Text ────────────────────────────────────────────────────────────

def build_semantic_text(title: str, also_called: list[str], topic_type: str,
                        body_systems: list[str], symptoms: list[str],
                        mesh_terms: list[str], summary: str,
                        related_topics: list[str]) -> str:
    """
    Build a single combined prose string optimized for embedding.
    This is what gets vectorized — must capture the "meaning" of the topic.
    """
    parts = []

    # Title and type
    parts.append(f"{title} is a {topic_type} topic.")

    # Aliases
    if also_called:
        parts.append(f"Also known as: {', '.join(also_called[:5])}.")

    # Body systems
    if body_systems and body_systems != ["General"]:
        parts.append(f"Body systems: {', '.join(body_systems)}.")

    # Symptoms (critical for symptom→condition matching)
    if symptoms:
        parts.append(f"Symptoms and signs: {', '.join(symptoms[:10])}.")

    # MeSH terms (medical vocabulary anchor)
    if mesh_terms:
        parts.append(f"Medical terms: {', '.join(mesh_terms[:8])}.")

    # Summary (first ~3 sentences for context)
    if summary:
        sentences = re.split(r"(?<=[.!?])\s+", summary)
        first_sentences = " ".join(sentences[:3])
        parts.append(first_sentences[:500])

    # Related topics
    if related_topics:
        parts.append(f"Related: {', '.join(related_topics[:5])}.")

    return " ".join(parts)


# ── Document ID ──────────────────────────────────────────────────────────────

def make_doc_id(title: str, topic_type: str) -> str:
    """Deterministic doc ID from title + type."""
    raw = f"{topic_type}:{title.lower()}"
    return f"nh_{topic_type[:4]}_{hashlib.md5(raw.encode()).hexdigest()[:10]}"


# ── Build Symptom-Condition Map ──────────────────────────────────────────────

def build_symptom_condition_map(docs: list[dict]) -> dict:
    """
    Build a map: symptom_phrase → [{condition, score, body_systems}]
    This powers the conversational agent's symptom→condition lookup.
    """
    symptom_map = defaultdict(list)

    for doc in docs:
        if doc["topic_type"] not in ("condition", "symptom"):
            continue

        title = doc["topic_name"]
        body_systems = doc["body_systems"]

        for symptom in doc["symptoms"]:
            symptom_map[symptom].append({
                "condition": title,
                "body_systems": body_systems,
                "topic_type": doc["topic_type"],
                "url": doc["url"],
            })

    # Sort conditions within each symptom by count of shared symptoms
    # (conditions with more listed symptoms are likely better documented)
    result = {}
    for symptom, conditions in sorted(symptom_map.items()):
        # Deduplicate
        seen = set()
        unique = []
        for c in conditions:
            if c["condition"] not in seen:
                seen.add(c["condition"])
                unique.append(c)
        result[symptom] = unique

    return result


# ── Build Keyword Lookup Map ─────────────────────────────────────────────────

def build_keyword_lookup_map(docs: list[dict]) -> dict:
    """
    Build flat map: any keyword/alias/synonym → canonical topic name.
    First-writer-wins for duplicates.
    """
    kwmap = {}

    for doc in docs:
        canonical = doc["topic_name"]

        # All keyword variations
        aliases = (
            [doc["topic_name"].lower()]
            + [ac.lower() for ac in doc.get("also_called", [])]
            + [sr.lower() for sr in doc.get("see_references", [])]
            + [m.lower() for m in doc.get("mesh_terms", [])]
        )

        for alias in aliases:
            alias = alias.strip()
            if alias and alias not in kwmap:
                kwmap[alias] = canonical

    return kwmap


# ── Process Definitions ──────────────────────────────────────────────────────

def transform_definitions(definitions: list[dict]) -> list[dict]:
    """Transform definition records into the unified document schema."""
    docs = []
    for d in definitions:
        if not d["term"] or not d["definition"]:
            continue

        title = d["term"]
        doc_id = make_doc_id(title, "definition")

        semantic_text = (
            f"{title} is a health definition in the category of {d['category']}. "
            f"Definition: {d['definition']}"
        )

        docs.append({
            "doc_id": doc_id,
            "topic_name": title,
            "topic_type": "definition",
            "also_called": [],
            "see_references": [],
            "body_systems": ["General"],
            "symptoms": [],
            "mesh_terms": [],
            "group_names": [d["category"]],
            "related_topic_names": [],
            "keywords": [title.lower(), d["category"].lower()],
            "summary": d["definition"][:1000],
            "semantic_text": semantic_text,
            "url": d.get("reference_url", ""),
            "source": "medlineplus_definitions",
            "module": "health_knowledge",
        })

    return docs


# ── Main Transform ───────────────────────────────────────────────────────────

def run():
    print("=" * 60)
    print("Step 2: Transform Health Topics into Unified Schema")
    print("=" * 60)

    # Load raw data
    with open(INPUT_TOPICS, encoding="utf-8") as f:
        topics = json.load(f)
    print(f"Loaded {len(topics)} raw health topics")

    with open(INPUT_DEFS, encoding="utf-8") as f:
        definitions = json.load(f)
    print(f"Loaded {len(definitions)} definitions")

    # Transform health topics
    semantic_docs = []
    type_counts = defaultdict(int)

    for t in tqdm(topics, desc="Transforming topics"):
        title = t["title"]
        summary = t["full_summary"]
        group_names = t["group_names"]
        mesh_terms = t["mesh_terms"]
        also_called = t["also_called"]
        see_refs = t.get("see_references", [])
        related_names = t.get("related_topic_names", [])

        # Classify topic type
        topic_type = classify_topic_type(title, group_names, summary)
        type_counts[topic_type] += 1

        # Infer body systems
        body_systems = infer_body_systems(group_names)

        # Extract symptoms
        symptoms = extract_symptoms(summary, title)

        # Build keywords
        keywords = build_keywords(
            title, also_called, see_refs, mesh_terms,
            group_names, symptoms, topic_type
        )

        # Build semantic text (for embedding)
        semantic_text = build_semantic_text(
            title, also_called, topic_type, body_systems,
            symptoms, mesh_terms, summary, related_names
        )

        doc_id = make_doc_id(title, topic_type)

        semantic_docs.append({
            "doc_id": doc_id,
            "topic_name": title,
            "topic_type": topic_type,
            "also_called": also_called,
            "see_references": see_refs,
            "body_systems": body_systems,
            "symptoms": symptoms,
            "mesh_terms": mesh_terms,
            "group_names": group_names,
            "related_topic_names": related_names,
            "keywords": keywords,
            "summary": summary[:1500],
            "semantic_text": semantic_text,
            "url": t["url"],
            "source": "medlineplus",
            "module": "health_knowledge",
        })

    # Transform definitions
    def_docs = transform_definitions(definitions)
    semantic_docs.extend(def_docs)
    print(f"Added {len(def_docs)} definition documents")

    # Build symptom→condition map
    symptom_map = build_symptom_condition_map(semantic_docs)
    print(f"Built symptom map: {len(symptom_map)} symptom phrases")

    # Build keyword lookup map
    keyword_map = build_keyword_lookup_map(semantic_docs)
    print(f"Built keyword map: {len(keyword_map)} entries")

    # Save outputs
    with open(OUT_SEMANTIC, "w", encoding="utf-8") as f:
        json.dump(semantic_docs, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {OUT_SEMANTIC} ({len(semantic_docs)} docs)")

    with open(OUT_SYMPTOM_MAP, "w", encoding="utf-8") as f:
        json.dump(symptom_map, f, indent=2, ensure_ascii=False)
    print(f"Saved: {OUT_SYMPTOM_MAP} ({len(symptom_map)} symptom entries)")

    with open(OUT_KEYWORD_MAP, "w", encoding="utf-8") as f:
        json.dump(keyword_map, f, indent=2, ensure_ascii=False)
    print(f"Saved: {OUT_KEYWORD_MAP} ({len(keyword_map)} keyword entries)")

    # Stats
    print(f"\n{'=' * 60}")
    print("Topic type distribution:")
    for tt, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {count:4d}  {tt}")

    # Sample symptom map entries
    print(f"\nSample symptom→condition mappings:")
    sample_symptoms = ["headache", "fever", "chest pain", "fatigue", "nausea"]
    for s in sample_symptoms:
        if s in symptom_map:
            conditions = [c["condition"] for c in symptom_map[s][:5]]
            print(f"  '{s}' → {conditions}")

    print(f"\nTotal documents: {len(semantic_docs)}")


if __name__ == "__main__":
    run()
