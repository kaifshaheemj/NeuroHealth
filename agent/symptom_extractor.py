import json

from llm.groq_client import call_groq_json
from agent.state import ConversationState

EXTRACTION_SYSTEM_PROMPT = """You are a medical symptom extraction system.
Given a patient's message and the current conversation context, extract:

1. "new_symptoms": List of symptom phrases mentioned (e.g., "headache", "nausea", "sensitivity to light")
2. "negated_symptoms": Symptoms the user explicitly denied having (e.g., if user says "no fever", extract "fever")
3. "symptom_details": Key-value details about known symptoms (e.g., {"headache": {"duration": "3 days", "severity": "severe", "location": "frontal"}})
4. "body_systems_mentioned": Any body regions or systems referenced (e.g., "head", "stomach", "chest")

Return ONLY valid JSON with these four keys. Do not add commentary.
If nothing is found for a key, return an empty list/object for it.

Normalize symptoms to simple medical terms:
- "my head is killing me" -> "headache"
- "feel like throwing up" -> "nausea"
- "can't sleep" -> "insomnia"
- "hurts when I pee" -> "painful urination"
- "feeling down" -> "depression"
- "heart is racing" -> "palpitations"
"""


def extract_symptoms(user_message: str, state: ConversationState) -> dict:
    """
    Extract symptoms from user message using LLM structured output.

    Returns: {
        "new_symptoms": ["headache", "nausea"],
        "negated_symptoms": ["fever"],
        "symptom_details": {"headache": {"duration": "3 days"}},
        "body_systems_mentioned": ["head"]
    }
    """
    context = ""
    if state.confirmed_symptoms:
        context = f"\nAlready known symptoms: {', '.join(state.confirmed_symptoms)}"
    if state.symptom_details:
        context += f"\nKnown details: {json.dumps(state.symptom_details)}"

    user_prompt = f"""Patient message: "{user_message}"
{context}

Extract symptoms, negations, details, and body systems from this message."""

    result = call_groq_json(
        system_prompt=EXTRACTION_SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )

    return {
        "new_symptoms": result.get("new_symptoms", []),
        "negated_symptoms": result.get("negated_symptoms", []),
        "symptom_details": result.get("symptom_details", {}),
        "body_systems_mentioned": result.get("body_systems_mentioned", []),
    }
