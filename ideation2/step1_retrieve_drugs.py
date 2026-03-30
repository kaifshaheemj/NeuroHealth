"""
Step 1: Retrieve ALL Drug Names from MedlinePlus Web Service
=============================================================
Uses the MedlinePlus Web Service documented at:
https://medlineplus.gov/about/developers/webservices/

Strategy:
  - Query: group:drugtherapy  (filters to all drug/medication topics)
  - rettype=all               (gets both brief snippet + full topic XML)
  - retmax=100                (100 per page, max allowed efficiently)
  - Paginate using retstart + file + server tokens (as per API docs)
  - Rate limit: 85 req/min → we use 0.8s delay = ~75 req/min (safe)

Base URL: https://wsearch.nlm.nih.gov/ws/query
No registration required. Free. Updated daily Tue–Sat.

Output: output/raw/medlineplus_drugs_raw.json
        (list of raw parsed XML records, one per drug topic)

Run: python step1_retrieve_drugs.py
"""

import requests
import xml.etree.ElementTree as ET
import json
import time
import re
import os
from tqdm import tqdm

BASE_URL   = "https://wsearch.nlm.nih.gov/ws/query"
OUTPUT     = "output/raw/medlineplus_drugs_raw.json"
BATCH_SIZE = 100       # records per API call (keep ≤100)
DELAY      = 0.8       # seconds between calls (75/min < 85/min limit)
TOOL_NAME  = "NeuroHealth-POC"
EMAIL      = "your@email.com"   # optional but recommended

os.makedirs("output/raw", exist_ok=True)


def strip_html(text: str) -> str:
    """Remove HTML tags and normalize whitespace."""
    if not text:
        return ""
    text = re.sub(r"<span[^>]*>(.*?)</span>", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_document(doc_el: ET.Element) -> dict:
    """
    Parse one <document> element from the MedlinePlus XML response.
    Extracts all content fields as documented in the API spec.
    """
    record = {
        "url":              doc_el.get("url", ""),
        "rank":             doc_el.get("rank", ""),
        "title":            "",
        "alt_titles":       [],    # brand names, synonyms, see-also refs
        "full_summary":     "",
        "mesh_terms":       [],    # Medical Subject Headings
        "group_names":      [],    # e.g. "Drug Therapy", "Endocrine System"
        "snippet":          "",
        "organization":     "",
    }

    for content in doc_el.findall("content"):
        name = content.attrib.get("name", "")
        raw  = ET.tostring(content, encoding="unicode", method="xml")
        # Extract inner text (strips XML tags)
        text = strip_html(raw.replace(f'<content name="{name}">', "").replace("</content>", ""))

        if   name == "title":            record["title"]        = text
        elif name == "altTitle":         record["alt_titles"].append(text)
        elif name == "FullSummary":      record["full_summary"] = text
        elif name == "mesh":             record["mesh_terms"].append(text)
        elif name == "groupName":        record["group_names"].append(text)
        elif name == "snippet":          record["snippet"]      = text
        elif name == "organizationName": record["organization"] = text

    return record


def fetch_page(term: str, retstart: int, file_token: str = None,
               server_token: str = None) -> tuple[list[dict], int, str, str]:
    """
    Fetch one page of results.
    Returns: (records, total_count, file_token, server_token)
    """
    params = {
        "db":      "healthTopics",
        "retmax":  str(BATCH_SIZE),
        "rettype": "all",          # full topic XML + brief snippets
        "tool":    TOOL_NAME,
        "email":   EMAIL,
    }

    if file_token and server_token:
        # Subsequent page — use pagination tokens from previous response
        params["file"]     = file_token
        params["server"]   = server_token
        params["retstart"] = str(retstart)
    else:
        # First page — use search term
        params["term"]     = term
        params["retstart"] = "0"

    resp = requests.get(BASE_URL, params=params, timeout=20)
    resp.raise_for_status()

    root  = ET.fromstring(resp.text)
    count = int(root.findtext("count") or "0")
    file_tok   = root.findtext("file")   or ""
    server_tok = root.findtext("server") or ""

    records = []
    for doc in root.findall(".//document"):
        parsed = parse_document(doc)
        # Only keep records that are actually drug-related
        groups = [g.lower() for g in parsed["group_names"]]
        if any("drug" in g or "medicine" in g or "medication" in g
               or "supplement" in g or "herb" in g for g in groups):
            records.append(parsed)
        elif parsed["title"]:
            # Include if title contains drug-class keywords
            title_lower = parsed["title"].lower()
            if any(kw in title_lower for kw in [
                "medicine", "drug", "medication", "tablet", "capsule",
                "injection", "vaccine", "supplement", "antibiotic",
                "inhibitor", "blocker", "agonist", "antagonist"
            ]):
                records.append(parsed)

    return records, count, file_tok, server_tok


def retrieve_all_drugs() -> list[dict]:
    """
    Paginate through ALL MedlinePlus drug topics using the Web Service.
    Uses the group:drugtherapy field filter to target drugs specifically.
    """
    print("Starting MedlinePlus drug retrieval...")
    print(f"Base URL: {BASE_URL}")
    print(f"Strategy: group:drugtherapy with rettype=all, paginating {BATCH_SIZE}/page\n")

    # Drug-focused query terms to maximize coverage
    # MedlinePlus groups drugs under these group names:
    DRUG_QUERIES = [
        "group:drugtherapy",           # primary drug therapy group
        "group:complementaryalternativemedicine",  # supplements/herbs
        "title:medicine",
        "title:drug",
        "title:medication",
        "title:antibiotic",
        "title:vaccine",
        "title:insulin",
        "title:supplement",
    ]

    all_records = []
    seen_urls   = set()

    for query_term in DRUG_QUERIES:
        print(f"\nQuerying: '{query_term}'")
        retstart    = 0
        file_tok    = None
        server_tok  = None
        total_count = None
        page        = 0

        while True:
            try:
                records, count, file_tok, server_tok = fetch_page(
                    term=query_term,
                    retstart=retstart,
                    file_token=file_tok,
                    server_token=server_tok,
                )
            except Exception as e:
                print(f"  [ERROR] page {page}: {e}")
                break

            if total_count is None:
                total_count = count
                pages_needed = (count + BATCH_SIZE - 1) // BATCH_SIZE
                print(f"  Total matching: {count} → {pages_needed} pages")

            # Deduplicate by URL
            new = [r for r in records if r["url"] not in seen_urls]
            for r in new:
                seen_urls.add(r["url"])
            all_records.extend(new)

            print(f"  Page {page+1}: +{len(new)} new records (total so far: {len(all_records)})")

            retstart += BATCH_SIZE
            page     += 1

            # Stop if we've retrieved all records or hit end
            if retstart >= total_count or not records or not file_tok:
                break

            time.sleep(DELAY)

        time.sleep(DELAY * 2)   # extra pause between different query terms

    print(f"\nTotal unique drug records retrieved: {len(all_records)}")
    return all_records


def run():
    records = retrieve_all_drugs()

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(records)} raw drug records → {OUTPUT}")

    # Quick sample preview
    print("\n--- Sample records ---")
    for r in records[:3]:
        print(f"  Drug:     {r['title']}")
        print(f"  AltNames: {r['alt_titles'][:4]}")
        print(f"  MeSH:     {r['mesh_terms'][:3]}")
        print(f"  Groups:   {r['group_names']}")
        print(f"  Summary:  {r['full_summary'][:120]}...")
        print()


if __name__ == "__main__":
    run()
