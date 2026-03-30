import json

from agent.state import ConversationState, ConversationPhase
from agent.intent_classifier import classify_and_extract
from agent.symptom_extractor import extract_symptoms
from agent.search_orchestrator import run_search, run_intent_search
from agent.prompts import (
    SYSTEM_PROMPT_BASE,
    FOLLOWUP_QUESTION_PROMPT,
    build_system_prompt_with_rag,
    INTENT_PROMPT_BUILDERS,
)
from llm.groq_client import call_groq_streaming
import config

# Simple greetings that should NOT trigger LLM classification
_GREETINGS = {
    "hi", "hello", "hey", "hii", "hiii", "yo", "sup",
    "good morning", "good afternoon", "good evening",
    "hi there", "hello there", "hey there",
}


def _is_greeting(text: str) -> bool:
    return text.strip().lower().rstrip("!.,") in _GREETINGS


def process_user_turn(user_message: str, state: ConversationState):
    """
    Main entry point for each user message.
    Yields response chunks (for Streamlit streaming) and mutates state in place.

    Flow: classify intent → route to appropriate handler.
    """
    state.add_message("user", user_message)

    # ── FAST PATH: trivial greetings (no LLM call) ──
    if _is_greeting(user_message) and state.phase in (
        ConversationPhase.GREETING, ConversationPhase.INITIAL_COMPLAINT
    ):
        state.phase = ConversationPhase.INITIAL_COMPLAINT
        yield from _generate_greeting_response(state)
        return

    # ── CLASSIFY INTENT ──
    classification = _safe_classify(user_message, state)
    _apply_classification(classification, state)
    intent = state.query_intent

    # ── ROUTE BY INTENT ──

    if intent == "greeting":
        state.phase = ConversationPhase.INITIAL_COMPLAINT
        yield from _generate_greeting_response(state)
        return

    if intent == "followup":
        yield from _handle_followup(user_message, state)
        return

    if intent in ("drug_query", "condition_info", "general_health"):
        yield from _handle_info_query(state, intent)
        return

    if intent == "symptom_report":
        yield from _handle_symptom_report(user_message, state, classification)
        return

    # ── FALLBACK ──
    yield from _handle_info_query(state, "general_health")


# ── Intent Classification ────────────────────────────────────────────────────

def _safe_classify(user_message: str, state: ConversationState) -> dict:
    """Run intent classification with error handling. Falls back to symptom extraction."""
    try:
        return classify_and_extract(user_message, state)
    except Exception as e:
        print(f"[WARN] Intent classification failed: {e}")
        # Fallback: try legacy symptom extraction
        try:
            extraction = extract_symptoms(user_message, state)
            extraction["intent"] = "symptom_report"
            extraction["entities"] = extraction.get("new_symptoms", [])
            return extraction
        except Exception:
            return {
                "intent": "general_health",
                "entities": [],
                "new_symptoms": [],
                "negated_symptoms": [],
                "symptom_details": {},
                "body_systems_mentioned": [],
            }


def _apply_classification(classification: dict, state: ConversationState):
    """Apply classified intent and extracted entities to state."""
    intent = classification.get("intent", "general_health")

    # Don't overwrite the previous intent if this is a followup
    if intent != "followup":
        state.query_intent = intent

    state.add_entities(classification.get("entities", []))

    # If it's a symptom report, also apply symptom-specific fields
    if intent == "symptom_report":
        state.add_symptoms(classification.get("new_symptoms", []))
        state.add_negated(classification.get("negated_symptoms", []))
        for symptom, details in classification.get("symptom_details", {}).items():
            if isinstance(details, dict):
                for key, value in details.items():
                    state.add_symptom_detail(symptom.lower(), key, str(value))


# ── Intent Handlers ──────────────────────────────────────────────────────────

def _handle_info_query(state: ConversationState, intent: str):
    """Handle drug_query, condition_info, and general_health intents."""
    state.phase = ConversationPhase.SEARCHING

    results = run_intent_search(state)
    state.search_results = results
    state.top_conditions = results[:config.TOP_CONDITIONS_SHOWN]

    state.phase = ConversationPhase.PRESENTING_RESULTS

    prompt_builder = INTENT_PROMPT_BUILDERS.get(intent, INTENT_PROMPT_BUILDERS["general_health"])
    system = prompt_builder(state, results)
    recent_messages = state.messages[-8:]

    full_response = ""
    for chunk in call_groq_streaming(system, recent_messages):
        full_response += chunk
        yield chunk

    state.add_message("assistant", full_response)


def _handle_followup(user_message: str, state: ConversationState):
    """Handle followup intent using cached results and previous intent's prompt."""
    state.phase = ConversationPhase.FOLLOW_UP

    if not state.search_results:
        results = run_intent_search(state)
        state.search_results = results

    # Use the previous intent's prompt builder
    previous_intent = state.query_intent if state.query_intent != "followup" else "general_health"
    prompt_builder = INTENT_PROMPT_BUILDERS.get(previous_intent, INTENT_PROMPT_BUILDERS["general_health"])
    system = prompt_builder(state, state.search_results)
    recent_messages = state.messages[-8:]

    full_response = ""
    for chunk in call_groq_streaming(system, recent_messages):
        full_response += chunk
        yield chunk

    state.add_message("assistant", full_response)


def _handle_symptom_report(user_message: str, state: ConversationState, classification: dict):
    """Handle symptom_report intent using the existing symptom pipeline."""

    # Early phases: start or continue symptom gathering
    if state.phase in (ConversationPhase.GREETING, ConversationPhase.INITIAL_COMPLAINT):
        state.initial_complaint_text = user_message

        if state.confirmed_symptoms:
            state.phase = ConversationPhase.GATHERING_DETAILS
            yield from _generate_followup(state)
        else:
            state.phase = ConversationPhase.INITIAL_COMPLAINT
            response = (
                "I want to make sure I understand what you're experiencing. "
                "Could you describe your main symptom or what's bothering you the most right now?"
            )
            state.add_message("assistant", response)
            yield response
        return

    if state.phase == ConversationPhase.GATHERING_DETAILS:
        state.followup_round += 1
        if state.should_search():
            yield from _execute_search_and_present(state)
        else:
            yield from _generate_followup(state)
        return

    if state.phase in (ConversationPhase.PRESENTING_RESULTS, ConversationPhase.FOLLOW_UP):
        state.phase = ConversationPhase.FOLLOW_UP
        if classification.get("new_symptoms"):
            # New symptoms added — re-search
            yield from _execute_search_and_present(state)
        else:
            yield from _generate_followup_answer(user_message, state)
        return

    # If we somehow got here from SEARCHING phase, run search
    yield from _execute_search_and_present(state)


# ── Shared Helpers ───────────────────────────────────────────────────────────

def _generate_greeting_response(state: ConversationState):
    """Generate initial greeting — no LLM call needed."""
    response = (
        "Hello! I'm NeuroHealth, your health information assistant. "
        "I can help you understand symptoms, look up conditions, "
        "find medication information, and answer general health questions.\n\n"
        "How can I help you today?"
    )
    state.add_message("assistant", response)
    yield response


def _generate_followup(state: ConversationState):
    """Generate a follow-up question using the LLM."""
    details_str = json.dumps(state.symptom_details) if state.symptom_details else "None yet"

    followup_instruction = FOLLOWUP_QUESTION_PROMPT.format(
        symptoms=", ".join(state.confirmed_symptoms),
        details=details_str,
        round=state.followup_round + 1,
        max_rounds=config.MAX_FOLLOWUP_ROUNDS,
    )

    system = SYSTEM_PROMPT_BASE + "\n\n" + followup_instruction
    recent_messages = state.messages[-6:]

    full_response = ""
    for chunk in call_groq_streaming(system, recent_messages):
        full_response += chunk
        yield chunk

    state.add_message("assistant", full_response)


def _execute_search_and_present(state: ConversationState):
    """Run ES search and generate response with RAG context."""
    state.phase = ConversationPhase.SEARCHING

    results = run_search(state)
    state.search_results = results
    state.top_conditions = results[:config.TOP_CONDITIONS_SHOWN]

    state.phase = ConversationPhase.PRESENTING_RESULTS

    system = build_system_prompt_with_rag(state, results)
    recent_messages = state.messages[-8:]

    full_response = ""
    for chunk in call_groq_streaming(system, recent_messages):
        full_response += chunk
        yield chunk

    state.add_message("assistant", full_response)


def _generate_followup_answer(user_message: str, state: ConversationState):
    """Answer a follow-up question using cached search results."""
    if state.search_results:
        system = build_system_prompt_with_rag(state, state.search_results)
    else:
        results = run_search(state)
        state.search_results = results
        system = build_system_prompt_with_rag(state, results)

    recent_messages = state.messages[-8:]

    full_response = ""
    for chunk in call_groq_streaming(system, recent_messages):
        full_response += chunk
        yield chunk

    state.add_message("assistant", full_response)
