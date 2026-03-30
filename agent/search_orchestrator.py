from core.es_client import (
    search_by_symptoms,
    search_health_topics,
    lookup_topic,
    get_related_conditions,
)
from agent.state import ConversationState
import config


def run_search(state: ConversationState) -> list[dict]:
    """
    Execute ES searches based on accumulated symptoms.
    Runs 3 strategies, merges, deduplicates, and ranks results.
    """
    symptoms = state.confirmed_symptoms
    if not symptoms:
        return []

    # Strategy 1: Symptom-map search (most precise)
    symptom_results = search_by_symptoms(symptoms, n=config.MAX_SEARCH_RESULTS)

    # Strategy 2: Natural-language hybrid search (broader)
    nl_query = _build_natural_query(state)
    hybrid_results = search_health_topics(nl_query, n=config.MAX_SEARCH_RESULTS)

    # Strategy 3: Direct topic lookup per symptom
    direct_matches = []
    for symptom in symptoms:
        canonical = lookup_topic(symptom)
        if canonical:
            topic_results = search_health_topics(canonical, n=2, topic_type="condition")
            direct_matches.extend(topic_results)

    # Merge and rank
    merged = _merge_results(symptom_results, hybrid_results, direct_matches)
    return merged[:config.TOP_CONDITIONS_SHOWN]


def get_condition_detail(condition_name: str) -> dict:
    """Fetch detailed info about a specific condition for follow-up."""
    results = search_health_topics(condition_name, n=1)
    related = get_related_conditions(condition_name, n=3)
    return {"primary": results[0] if results else {}, "related": related}


def _build_natural_query(state: ConversationState) -> str:
    """Build a natural-language query string from symptoms and their details."""
    parts = []
    for symptom in state.confirmed_symptoms:
        detail = state.symptom_details.get(symptom, {})
        desc = symptom
        if "duration" in detail:
            desc += f" for {detail['duration']}"
        if "severity" in detail:
            desc += f" ({detail['severity']})"
        if "location" in detail:
            desc += f" in {detail['location']}"
        parts.append(desc)
    return ", ".join(parts)


def run_intent_search(state: ConversationState) -> list[dict]:
    """
    Dispatch search based on the classified intent.
    For symptom_report, delegates to the existing run_search().
    For other intents, calls the appropriate search strategy.
    """
    intent = state.query_intent

    if intent == "symptom_report":
        return run_search(state)
    if intent == "drug_query":
        return _search_drug(state)
    if intent == "condition_info":
        return _search_condition(state)
    # general_health or any fallback
    return _search_general(state)


def _search_drug(state: ConversationState) -> list[dict]:
    """Search for drug/medication information."""
    entities = state.mentioned_entities
    if not entities:
        return []

    all_results = []
    for entity in entities:
        canonical = lookup_topic(entity)
        query = canonical if canonical else entity

        results = search_health_topics(query, n=config.MAX_SEARCH_RESULTS, topic_type="drug")
        all_results.extend(results)

        # If no drug-specific results, try a broader search
        if not results:
            results = search_health_topics(query, n=config.MAX_SEARCH_RESULTS)
            all_results.extend(results)

    seen = set()
    deduped = []
    for r in all_results:
        name = r.get("topic_name", "")
        if name and name not in seen:
            seen.add(name)
            deduped.append(r)

    state.last_search_query = ", ".join(entities)
    return deduped[:config.TOP_CONDITIONS_SHOWN]


def _search_condition(state: ConversationState) -> list[dict]:
    """Search for condition/disease information."""
    entities = state.mentioned_entities
    if not entities:
        return []

    all_results = []
    for entity in entities:
        canonical = lookup_topic(entity)
        query = canonical if canonical else entity

        results = search_health_topics(query, n=config.MAX_SEARCH_RESULTS, topic_type="condition")
        all_results.extend(results)

        related = get_related_conditions(query, n=3)
        for r in related:
            if isinstance(r, dict):
                all_results.append(r)

    seen = set()
    deduped = []
    for r in all_results:
        name = r.get("topic_name", "")
        if name and name not in seen:
            seen.add(name)
            deduped.append(r)

    state.last_search_query = ", ".join(entities)
    return deduped[:config.TOP_CONDITIONS_SHOWN]


def _search_general(state: ConversationState) -> list[dict]:
    """General health topic search, no type filter."""
    entities = state.mentioned_entities
    query = ", ".join(entities) if entities else state.initial_complaint_text

    if not query:
        return []

    results = search_health_topics(query, n=config.MAX_SEARCH_RESULTS)
    state.last_search_query = query
    return results[:config.TOP_CONDITIONS_SHOWN]


def _merge_results(
    symptom_results: list[dict],
    hybrid_results: list[dict],
    direct_results: list[dict],
) -> list[dict]:
    """
    Merge results from 3 strategies with weighted scoring.
    Symptom-map results get highest weight (directly match symptom→condition).
    """
    scored = {}

    # Symptom map results (weight 3x — most precise)
    for r in symptom_results:
        name = r.get("condition", "")
        if not name:
            continue
        if name not in scored:
            scored[name] = {"data": r, "total_score": 0, "sources": set()}
        scored[name]["total_score"] += r.get("score", 1) * 3.0
        scored[name]["sources"].add("symptom_map")

    # Hybrid search results (weight 2x)
    for r in hybrid_results:
        name = r.get("topic_name", "")
        if not name:
            continue
        if name not in scored:
            scored[name] = {"data": r, "total_score": 0, "sources": set()}
        scored[name]["total_score"] += r.get("score", 1) * 2.0
        scored[name]["sources"].add("hybrid")

    # Direct lookup results (weight 1.5x)
    for r in direct_results:
        name = r.get("topic_name", "")
        if not name:
            continue
        if name not in scored:
            scored[name] = {"data": r, "total_score": 0, "sources": set()}
        scored[name]["total_score"] += r.get("score", 1) * 1.5
        scored[name]["sources"].add("direct")

    # Bonus for appearing in multiple strategies
    for entry in scored.values():
        entry["total_score"] += (len(entry["sources"]) - 1) * 2.0

    # Sort by total score descending
    ranked = sorted(scored.values(), key=lambda x: -x["total_score"])

    results = []
    for entry in ranked:
        data = entry["data"]
        data["final_score"] = round(entry["total_score"], 3)
        data["search_sources"] = list(entry["sources"])
        results.append(data)

    return results
