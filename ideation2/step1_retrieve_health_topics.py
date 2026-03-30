"""
Step 1: Retrieve ALL MedlinePlus Health Topics (Bulk XML Download)
==================================================================
Downloads the complete MedlinePlus dataset:

1. Health Topics XML  — ~1000+ topics (conditions, symptoms, drugs, wellness)
2. Topic Groups XML   — body-system / category groupings
3. Definitions XMLs   — fitness, nutrition, vitamins, minerals, general health

Uses the official bulk XML files (updated Tue-Sat) instead of the
search API — guarantees 100% coverage in a single download.

Output:
  output/raw/health_topics_raw.json
  output/raw/topic_groups.json
  output/raw/definitions_raw.json

Run: python step1_retrieve_health_topics.py
"""

import json
import os
import re
import time
import xml.etree.ElementTree as ET

import requests
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────

# Bulk XML URLs (date rotates daily Tue-Sat, we fetch the index page to find latest)
BASE_XML_URL = "https://medlineplus.gov/xml"
TOPICS_XML_URL = f"{BASE_XML_URL}/mplus_topics_2026-03-19.xml"
GROUPS_XML_URL = f"{BASE_XML_URL}/mplus_topic_groups_2026-03-19.xml"

DEFINITION_URLS = [
    f"{BASE_XML_URL}/fitnessdefinitions.xml",
    f"{BASE_XML_URL}/generalhealthdefinitions.xml",
    f"{BASE_XML_URL}/mineralsdefinitions.xml",
    f"{BASE_XML_URL}/nutritiondefinitions.xml",
    f"{BASE_XML_URL}/vitaminsdefinitions.xml",
]

OUT_DIR = "output/raw"
OUT_TOPICS = os.path.join(OUT_DIR, "health_topics_raw.json")
OUT_GROUPS = os.path.join(OUT_DIR, "topic_groups.json")
OUT_DEFS = os.path.join(OUT_DIR, "definitions_raw.json")

os.makedirs(OUT_DIR, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def strip_html(text: str) -> str:
    """Remove HTML tags and collapse whitespace."""
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def download_xml(url: str, label: str) -> ET.Element:
    """Download an XML file and return the parsed root element."""
    print(f"Downloading {label}...")
    print(f"  URL: {url}")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    size_mb = len(resp.content) / (1024 * 1024)
    print(f"  Downloaded: {size_mb:.1f} MB")
    return ET.fromstring(resp.content)


def find_latest_topics_url() -> str:
    """
    Try to find the latest health topics XML URL.
    Falls back to a known date if date detection fails.
    """
    from datetime import datetime, timedelta

    # Try last 7 days (XML published Tue-Sat)
    for days_back in range(7):
        d = datetime.now() - timedelta(days=days_back)
        date_str = d.strftime("%Y-%m-%d")
        url = f"{BASE_XML_URL}/mplus_topics_{date_str}.xml"
        try:
            resp = requests.head(url, timeout=10)
            if resp.status_code == 200:
                return url, date_str
        except requests.RequestException:
            continue

    # Fallback
    return TOPICS_XML_URL, date_str


def find_latest_groups_url(date_str: str) -> str:
    """Get groups XML URL for the same date as topics."""
    return f"{BASE_XML_URL}/mplus_topic_groups_{date_str}.xml"


# ── Parse Health Topics ──────────────────────────────────────────────────────

def parse_health_topics(root: ET.Element) -> list[dict]:
    """
    Parse the bulk health topics XML into structured records.
    Extracts ALL topics — conditions, symptoms, drugs, wellness, etc.
    """
    topics = []
    total = root.get("total", "?")
    print(f"Parsing {total} health topics...")

    for ht in tqdm(root.findall("health-topic"), desc="Parsing topics"):
        # Skip Spanish topics (we'll handle English only for now)
        if ht.get("language", "English") != "English":
            continue

        topic_id = ht.get("id", "")
        title = ht.get("title", "")
        url = ht.get("url", "")
        meta_desc = ht.get("meta-desc", "")
        date_created = ht.get("date-created", "")

        # Also-called / synonyms
        also_called = [ac.text.strip() for ac in ht.findall("also-called")
                       if ac.text and ac.text.strip()]

        # Full summary (HTML content)
        full_summary_el = ht.find("full-summary")
        full_summary_raw = full_summary_el.text if full_summary_el is not None and full_summary_el.text else ""
        full_summary = strip_html(full_summary_raw)

        # Groups (body system / category)
        groups = []
        for g in ht.findall("group"):
            groups.append({
                "id": g.get("id", ""),
                "name": g.text.strip() if g.text else "",
                "url": g.get("url", ""),
            })

        # MeSH headings
        mesh_headings = []
        for mh in ht.findall("mesh-heading"):
            desc_el = mh.find("descriptor")
            if desc_el is not None and desc_el.text:
                descriptor = desc_el.text.strip()
                qualifiers = []
                for q in mh.findall("qualifier"):
                    if q.text:
                        qualifiers.append(q.text.strip())
                mesh_headings.append({
                    "descriptor": descriptor,
                    "descriptor_id": desc_el.get("id", ""),
                    "qualifiers": qualifiers,
                })

        # Related topics
        related_topics = []
        for rt in ht.findall("related-topic"):
            if rt.text:
                related_topics.append({
                    "id": rt.get("id", ""),
                    "name": rt.text.strip(),
                    "url": rt.get("url", ""),
                })

        # See references (alternate names that redirect)
        see_references = [sr.text.strip() for sr in ht.findall("see-reference")
                          if sr.text and sr.text.strip()]

        # Primary institute
        pi_el = ht.find("primary-institute")
        primary_institute = ""
        if pi_el is not None and pi_el.text:
            primary_institute = pi_el.text.strip()

        # Sites (external resources linked from the topic)
        sites = []
        for site in ht.findall("site"):
            site_title = site.get("title", "")
            site_url = site.get("url", "")
            categories = [ic.text.strip() for ic in site.findall("information-category")
                          if ic.text]
            orgs = [o.text.strip() for o in site.findall("organization") if o.text]
            sites.append({
                "title": site_title,
                "url": site_url,
                "categories": categories,
                "organizations": orgs,
            })

        topics.append({
            "topic_id": topic_id,
            "title": title,
            "url": url,
            "meta_desc": meta_desc,
            "date_created": date_created,
            "also_called": also_called,
            "see_references": see_references,
            "full_summary": full_summary,
            "full_summary_html": full_summary_raw,
            "groups": groups,
            "group_names": [g["name"] for g in groups],
            "mesh_headings": mesh_headings,
            "mesh_terms": [mh["descriptor"] for mh in mesh_headings],
            "related_topics": related_topics,
            "related_topic_names": [rt["name"] for rt in related_topics],
            "primary_institute": primary_institute,
            "sites": sites,
            "site_count": len(sites),
        })

    return topics


# ── Parse Topic Groups ───────────────────────────────────────────────────────

def parse_topic_groups(root: ET.Element) -> list[dict]:
    """Parse the topic groups XML into a list of group records."""
    groups = []
    for g in root.findall("group"):
        if g.get("language", "English") != "English":
            continue
        groups.append({
            "id": g.get("id", ""),
            "name": g.text.strip() if g.text else "",
            "url": g.get("url", ""),
        })
    print(f"Parsed {len(groups)} topic groups")
    return groups


# ── Parse Definitions ────────────────────────────────────────────────────────

def parse_definitions(url: str) -> list[dict]:
    """Download and parse a definitions XML file."""
    try:
        root = download_xml(url, os.path.basename(url))
    except Exception as e:
        print(f"  [WARN] Failed to download {url}: {e}")
        return []

    category = root.get("title", os.path.basename(url).replace("definitions.xml", ""))
    defs = []
    for tg in root.findall("term-group"):
        term_el = tg.find("term")
        def_el = tg.find("definition")
        if term_el is not None and def_el is not None:
            defs.append({
                "term": term_el.text.strip() if term_el.text else "",
                "definition": def_el.text.strip() if def_el.text else "",
                "reference": tg.get("reference", ""),
                "reference_url": tg.get("reference-url", ""),
                "category": category,
            })

    print(f"  Parsed {len(defs)} definitions from {category}")
    return defs


# ── Main ─────────────────────────────────────────────────────────────────────

def run():
    print("=" * 60)
    print("Step 1: Retrieve ALL MedlinePlus Health Data")
    print("=" * 60)

    # 1. Find latest XML URLs
    topics_url, date_str = find_latest_topics_url()
    groups_url = find_latest_groups_url(date_str)
    print(f"Using XML date: {date_str}\n")

    # 2. Download and parse health topics (the big one ~29MB)
    topics_root = download_xml(topics_url, "Health Topics XML")
    topics = parse_health_topics(topics_root)
    print(f"  Total English topics: {len(topics)}")

    # 3. Download and parse topic groups
    try:
        groups_root = download_xml(groups_url, "Topic Groups XML")
        groups = parse_topic_groups(groups_root)
    except Exception as e:
        print(f"  [WARN] Groups download failed: {e}")
        # Derive groups from topics as fallback
        group_map = {}
        for t in topics:
            for g in t["groups"]:
                if g["id"] not in group_map:
                    group_map[g["id"]] = g
        groups = list(group_map.values())
        print(f"  Derived {len(groups)} groups from topics")

    # 4. Download and parse all definition files
    all_definitions = []
    print("\nDownloading definitions...")
    for def_url in DEFINITION_URLS:
        defs = parse_definitions(def_url)
        all_definitions.extend(defs)
    print(f"Total definitions: {len(all_definitions)}")

    # 5. Save outputs
    with open(OUT_TOPICS, "w", encoding="utf-8") as f:
        json.dump(topics, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {OUT_TOPICS} ({len(topics)} topics)")

    with open(OUT_GROUPS, "w", encoding="utf-8") as f:
        json.dump(groups, f, indent=2, ensure_ascii=False)
    print(f"Saved: {OUT_GROUPS} ({len(groups)} groups)")

    with open(OUT_DEFS, "w", encoding="utf-8") as f:
        json.dump(all_definitions, f, indent=2, ensure_ascii=False)
    print(f"Saved: {OUT_DEFS} ({len(all_definitions)} definitions)")

    # Stats
    print(f"\n{'=' * 60}")
    print("Summary:")
    print(f"  Health topics:  {len(topics)}")
    print(f"  Topic groups:   {len(groups)}")
    print(f"  Definitions:    {len(all_definitions)}")

    group_counts = {}
    for t in topics:
        for gn in t["group_names"]:
            group_counts[gn] = group_counts.get(gn, 0) + 1
    print(f"\nTop groups by topic count:")
    for gn, count in sorted(group_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"  {count:4d}  {gn}")


if __name__ == "__main__":
    run()
