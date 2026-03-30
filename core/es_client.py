import sys
import os

from elasticsearch import Elasticsearch

import config

# Add the ideation2 directory to path so we can import the query functions
_ideation2_path = os.path.join(os.path.dirname(__file__), "..", "Preparation", "ideation2")
if _ideation2_path not in sys.path:
    sys.path.insert(0, os.path.abspath(_ideation2_path))

_es_instance = None
_query_funcs = None


def get_es() -> Elasticsearch:
    """Singleton Elasticsearch connection."""
    global _es_instance
    if _es_instance is None:
        _es_instance = Elasticsearch(
            config.ES_HOST,
            basic_auth=(config.ES_USER, config.ES_PASS),
            verify_certs=config.ES_VERIFY_CERTS,
        )
        if not _es_instance.ping():
            raise ConnectionError("Cannot reach Elasticsearch")
    return _es_instance


def _load_query_funcs():
    """Lazy-load query functions from step3 to avoid eagerly loading the embedding model."""
    global _query_funcs
    if _query_funcs is None:
        from step3_ingest_health_topics import (
            search_by_symptoms as _search_by_symptoms,
            search_health_topics as _search_health_topics,
            lookup_topic as _lookup_topic,
            get_related_conditions as _get_related_conditions,
            autocomplete_topic as _autocomplete_topic,
        )
        _query_funcs = {
            "search_by_symptoms": _search_by_symptoms,
            "search_health_topics": _search_health_topics,
            "lookup_topic": _lookup_topic,
            "get_related_conditions": _get_related_conditions,
            "autocomplete_topic": _autocomplete_topic,
        }
    return _query_funcs


def search_by_symptoms(symptoms: list[str], n: int = 10) -> list[dict]:
    """Find conditions matching a list of user-reported symptoms."""
    funcs = _load_query_funcs()
    return funcs["search_by_symptoms"](get_es(), symptoms, n)


def search_health_topics(query: str, n: int = 5,
                         topic_type: str = None,
                         body_system: str = None) -> list[dict]:
    """Hybrid BM25 + semantic search across all health topics."""
    funcs = _load_query_funcs()
    return funcs["search_health_topics"](get_es(), query, n, topic_type, body_system)


def lookup_topic(user_term: str) -> str | None:
    """O(1) alias resolution: any synonym -> canonical topic name."""
    funcs = _load_query_funcs()
    return funcs["lookup_topic"](get_es(), user_term)


def get_related_conditions(topic_name: str, n: int = 5) -> list[dict]:
    """Find conditions related to a given topic."""
    funcs = _load_query_funcs()
    return funcs["get_related_conditions"](get_es(), topic_name, n)


def autocomplete_topic(prefix: str, n: int = 5) -> list[str]:
    """Autocomplete health topic name."""
    funcs = _load_query_funcs()
    return funcs["autocomplete_topic"](get_es(), prefix, n)
