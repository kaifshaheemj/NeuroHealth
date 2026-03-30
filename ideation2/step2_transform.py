"""
Step 2: Transform Raw Records → Semantic JSON + Keyword Map
=============================================================
Takes raw output from step1 and produces two things:

A) Semantic JSON document (one per drug) — for embedding + ES ingestion
B) Keyword map JSON — flat lookup dict: any keyword/alias → canonical drug name

Semantic JSON structure (one per drug):
{
  "doc_id":           "nh_drug_abc123",
  "drug_name":        "Metformin",               ← canonical name
  "brand_names":      ["Glucophage", "Fortamet"],
  "drug_class":       "Biguanide antidiabetic",
  "generic_name":     "metformin hydrochloride",
  "mesh_terms":       ["Metformin", "Hypoglycemic Agents"],
  "uses":             ["type 2 diabetes", "blood sugar control"],
  "keywords":         ["metformin", "glucophage", "diabetes medicine", ...],
  "summary":          "Metformin is used to treat...",
  "semantic_text":    "Metformin (Glucophage) is a diabetes medicine...",  ← for embedding
  "url":              "https://medlineplus.gov/druginfo/meds/a696005.html",
  "source":           "medlineplus_webservice",
  "module":           "medications",
}

Keyword map (flat dict):
{
  "metformin":          "Metformin",
  "glucophage":         "Metformin",
  "fortamet":           "Metformin",
  "diabetes medicine":  "Metformin",
  "biguanide":          "Metformin",
  ...
}

Output:
  output/processed/drugs_semantic.json
  output/processed/drugs_keyword_map.json

Run: python step2_transform.py
"""

import json
import re
import hashlib
import os
from tqdm import tqdm

INPUT  = "output/raw/medlineplus_drugs_raw.json"
OUT_SEMANTIC = "output/processed/drugs_semantic.json"
OUT_KWMAP    = "output/processed/drugs_keyword_map.json"

os.makedirs("output/processed", exist_ok=True)


# ── Drug class inference from MeSH + title ───────────────────────────────────

DRUG_CLASS_PATTERNS = [
    (r"antibiotic|antimicrobial|penicillin|cephalosporin|macrolide|fluoroquinolone",
     "Antibiotic"),
    (r"antiviral|antifungal|antiparasitic",
     "Antimicrobial"),
    (r"antidepressant|ssri|snri|tricyclic|maoi",
     "Antidepressant"),
    (r"antihypertensive|ace inhibitor|arb|beta.blocker|calcium channel",
     "Antihypertensive"),
    (r"antidiabetic|hypoglycemic|insulin|metformin|glipizide|glimepiride",
     "Antidiabetic"),
    (r"analgesic|pain relief|nsaid|opioid|painkiller",
     "Analgesic / Pain Relief"),
    (r"anticoagulant|blood thinner|warfarin|heparin|antiplatelet",
     "Anticoagulant"),
    (r"statin|cholesterol|lipid.lowering|atorvastatin|simvastatin",
     "Lipid-Lowering Agent"),
    (r"antipsychotic|neuroleptic|antischizophrenic",
     "Antipsychotic"),
    (r"anticonvulsant|antiepileptic|seizure",
     "Anticonvulsant"),
    (r"bronchodilator|inhaler|asthma|copd|salbutamol|albuterol",
     "Bronchodilator"),
    (r"antihistamine|allergy|antiallergic",
     "Antihistamine"),
    (r"proton pump|antacid|ppi|omeprazole|pantoprazole",
     "Gastrointestinal Agent"),
    (r"hormone|corticosteroid|steroid|thyroid|estrogen|testosterone",
     "Hormonal Agent"),
    (r"vaccine|immunization|immunoglobulin",
     "Vaccine / Immunization"),
    (r"vitamin|mineral|supplement|probiotic|herbal",
     "Supplement / Vitamin"),
    (r"chemotherapy|antineoplastic|anticancer|cytotoxic",
     "Antineoplastic"),
    (r"diuretic|water pill|furosemide|hydrochlorothiazide",
     "Diuretic"),
    (r"antifungal|fluconazole|itraconazole|clotrimazole",
     "Antifungal"),
    (r"muscle relaxant|neuromuscular",
     "Muscle Relaxant"),
]

def infer_drug_class(title: str, mesh_terms: list, summary: str) -> str:
    combined = f"{title} {' '.join(mesh_terms)} {summary[:300]}".lower()
    for pattern, drug_class in DRUG_CLASS_PATTERNS:
        if re.search(pattern, combined, re.IGNORECASE):
            return drug_class
    return "General Medication"


def extract_uses(summary: str, mesh_terms: list) -> list[str]:
    """
    Extract conditions/uses this drug treats from the summary text.
    Simple heuristic: sentences containing 'used to treat', 'used for', 'treats'.
    """
    uses = []
    patterns = [
        r"used (?:to treat|for) ([^.;,]{5,60})",
        r"treats? ([^.;,]{5,60})",
        r"prescribed (?:for|to treat) ([^.;,]{5,60})",
        r"indicated for ([^.;,]{5,60})",
        r"treatment of ([^.;,]{5,60})",
    ]
    for pat in patterns:
        matches = re.findall(pat, summary, re.IGNORECASE)
        uses.extend([m.strip().lower() for m in matches if len(m.strip()) > 4])

    # Add MeSH terms as use hints (they often describe the indication)
    for mesh in mesh_terms:
        if not any(kw in mesh.lower() for kw in ["agent", "drug", "compound", "therapy"]):
            uses.append(mesh.lower())

    # Deduplicate and limit
    seen, deduped = set(), []
    for u in uses:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped[:10]


def build_keywords(drug_name: str, alt_titles: list, mesh_terms: list,
                   drug_class: str, uses: list) -> list[str]:
    """
    Build exhaustive keyword list for this drug.
    Every keyword in this list → this drug in the keyword map.
    """
    kws = set()

    # Canonical name and all its variations
    kws.add(drug_name.lower())
    kws.add(drug_name.lower().strip())

    # Alt titles (brand names, synonyms, see-also)
    for alt in alt_titles:
        cleaned = re.sub(r"\s+", " ", alt.strip().lower())
        if 2 < len(cleaned) < 80:
            kws.add(cleaned)

    # MeSH terms
    for mesh in mesh_terms:
        kws.add(mesh.lower())

    # Drug class
    if drug_class and drug_class != "General Medication":
        kws.add(drug_class.lower())

    # Common colloquial phrasings
    kws.add(f"{drug_name.lower()} tablet")
    kws.add(f"{drug_name.lower()} capsule")
    kws.add(f"{drug_name.lower()} medicine")
    kws.add(f"{drug_name.lower()} drug")
    kws.add(f"{drug_name.lower()} medication")

    # Uses as keywords (e.g. "diabetes medicine" → Metformin)
    for use in uses[:3]:
        kws.add(f"{use} medicine")
        kws.add(f"{use} drug")
        kws.add(f"{use} medication")

    # Remove empty/too-short
    return sorted([k for k in kws if len(k) > 1])


def build_semantic_text(drug_name: str, alt_titles: list, drug_class: str,
                        uses: list, summary: str) -> str:
    """
    Build the text that will be embedded as a dense vector.
    Should capture: name, aliases, class, what it's for, key facts.
    """
    parts = []

    brand_str = ", ".join(alt_titles[:4]) if alt_titles else ""
    if brand_str:
        parts.append(f"{drug_name} (also known as {brand_str}) is a {drug_class}.")
    else:
        parts.append(f"{drug_name} is a {drug_class}.")

    if uses:
        uses_str = ", ".join(uses[:4])
        parts.append(f"It is used for: {uses_str}.")

    if summary:
        # Take first 3 sentences of summary
        sentences = re.split(r"(?<=[.!?])\s+", summary)
        parts.append(" ".join(sentences[:3]))

    return " ".join(parts)


def make_doc_id(drug_name: str) -> str:
    return "nh_drug_" + hashlib.md5(drug_name.lower().encode()).hexdigest()[:10]


def transform(raw_records: list[dict]) -> tuple[list[dict], dict]:
    """
    Transform raw parsed records into semantic docs + keyword map.
    Returns: (semantic_docs, keyword_map)
    """
    semantic_docs = []
    keyword_map   = {}   # keyword (lowercase) → canonical drug name

    print(f"Transforming {len(raw_records)} raw records...")

    for raw in tqdm(raw_records):
        title = raw.get("title", "").strip()
        if not title:
            continue

        alt_titles  = raw.get("alt_titles", [])
        mesh_terms  = raw.get("mesh_terms", [])
        group_names = raw.get("group_names", [])
        summary     = raw.get("full_summary", "") or raw.get("snippet", "")
        url         = raw.get("url", "")

        drug_class = infer_drug_class(title, mesh_terms, summary)
        uses       = extract_uses(summary, mesh_terms)
        keywords   = build_keywords(title, alt_titles, mesh_terms, drug_class, uses)
        sem_text   = build_semantic_text(title, alt_titles, drug_class, uses, summary)

        # Separate brand names from see-also refs (heuristic: shorter = brand name)
        brand_names = [a for a in alt_titles if len(a.split()) <= 3]
        see_also    = [a for a in alt_titles if len(a.split()) > 3]

        doc = {
            "doc_id":         make_doc_id(title),
            "drug_name":      title,
            "brand_names":    brand_names,
            "see_also":       see_also,
            "generic_name":   title.lower(),
            "drug_class":     drug_class,
            "mesh_terms":     mesh_terms,
            "group_names":    group_names,
            "uses":           uses,
            "keywords":       keywords,
            "summary":        summary[:1000],
            "semantic_text":  sem_text,    # this field gets embedded
            "url":            url,
            "source":         "medlineplus_webservice",
            "module":         "medications",
        }
        semantic_docs.append(doc)

        # Build keyword map entries
        for kw in keywords:
            if kw and kw not in keyword_map:
                keyword_map[kw] = title

    print(f"Produced: {len(semantic_docs)} semantic docs")
    print(f"Keyword map entries: {len(keyword_map)}")
    return semantic_docs, keyword_map


def run():
    with open(INPUT, encoding="utf-8") as f:
        raw_records = json.load(f)

    semantic_docs, keyword_map = transform(raw_records)

    with open(OUT_SEMANTIC, "w", encoding="utf-8") as f:
        json.dump(semantic_docs, f, indent=2, ensure_ascii=False)

    with open(OUT_KWMAP, "w", encoding="utf-8") as f:
        json.dump(keyword_map, f, indent=2, ensure_ascii=False, sort_keys=True)

    print(f"\nSaved:")
    print(f"  Semantic docs  → {OUT_SEMANTIC}")
    print(f"  Keyword map    → {OUT_KWMAP}")

    # Preview
    print("\n--- Sample semantic doc ---")
    if semantic_docs:
        d = semantic_docs[0]
        print(json.dumps({
            "drug_name":     d["drug_name"],
            "brand_names":   d["brand_names"][:3],
            "drug_class":    d["drug_class"],
            "uses":          d["uses"][:3],
            "keywords":      d["keywords"][:6],
            "semantic_text": d["semantic_text"][:200],
        }, indent=2))

    print("\n--- Sample keyword map entries ---")
    for k, v in list(keyword_map.items())[:8]:
        print(f"  '{k}' → '{v}'")


if __name__ == "__main__":
    run()
