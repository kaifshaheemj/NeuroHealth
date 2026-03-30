import json

from llm.groq_client import call_groq_json
from agent.state import ConversationState

INTENT_CLASSIFICATION_PROMPT = """You are an intent classifier for a health information assistant.
Given a user message and conversation context, determine what the user wants and extract relevant entities.

INTENT TYPES (pick exactly one):
- "symptom_report": User is describing physical/mental symptoms they are experiencing.
  Examples: "I have a headache", "My stomach hurts", "I've been feeling dizzy for 3 days"
- "drug_query": User is asking about a medication, drug, supplement, or treatment.
  Examples: "Is ibuprofen safe for children?", "What are the side effects of metformin?", "Can I take aspirin with blood thinners?"
- "condition_info": User is asking what a medical condition or disease is.
  Examples: "What is diabetes?", "Tell me about asthma", "How does arthritis develop?"
- "general_health": User is asking a general wellness or health question not about a specific drug or condition diagnosis.
  Examples: "How to improve sleep?", "What foods lower cholesterol?", "Tips for managing stress"
- "greeting": User is saying hello or making small talk with no health question.
  Examples: "Hello", "Hi there", "Good morning"
- "followup": User is asking a follow-up about something already discussed in the conversation.
  Examples: "Tell me more about that", "What about the side effects?", "Can you explain the second one?"

ENTITY EXTRACTION:
Also extract entities mentioned in the message:
- For symptom_report: extract symptoms as normalized medical terms
- For drug_query: extract drug/medication names mentioned
- For condition_info: extract condition/disease names mentioned
- For general_health: extract the health topic keywords
- For greeting/followup: entities can be empty

SYMPTOM DETAILS (only when intent is "symptom_report"):
If the user reports symptoms, also extract:
- "negated_symptoms": symptoms explicitly denied ("no fever" -> "fever")
- "symptom_details": details about symptoms (duration, severity, location, triggers)
- "body_systems_mentioned": body regions referenced

Return ONLY valid JSON with these keys:
{
    "intent": "<one of the 6 intent types>",
    "entities": ["entity1", "entity2"],
    "new_symptoms": [],
    "negated_symptoms": [],
    "symptom_details": {},
    "body_systems_mentioned": []
}

If the message is ambiguous, prefer the most specific intent (drug_query > general_health, condition_info > general_health).
Normalize symptom terms: "my head is killing me" -> "headache", "can't sleep" -> "insomnia".
"""


def classify_and_extract(user_message: str, state: ConversationState) -> dict:
    """
    Classify user intent and extract entities in a single LLM call.

    Returns:
        {
            "intent": str,          # one of: symptom_report, drug_query, condition_info,
                                    #         general_health, greeting, followup
            "entities": list[str],  # drug names, condition names, symptoms, or topic keywords
            "new_symptoms": list[str],
            "negated_symptoms": list[str],
            "symptom_details": dict,
            "body_systems_mentioned": list[str],
        }
    """
    context_parts = []
    if state.confirmed_symptoms:
        context_parts.append(f"Already known symptoms: {', '.join(state.confirmed_symptoms)}")
    if state.symptom_details:
        context_parts.append(f"Known details: {json.dumps(state.symptom_details)}")
    if state.query_intent:
        context_parts.append(f"Previous intent: {state.query_intent}")
    if state.mentioned_entities:
        context_parts.append(f"Previously mentioned entities: {', '.join(state.mentioned_entities)}")

    context = "\n".join(context_parts) if context_parts else "No prior context."

    user_prompt = f"""User message: "{user_message}"

Conversation context:
{context}

Classify the intent and extract entities."""

    result = call_groq_json(
        system_prompt=INTENT_CLASSIFICATION_PROMPT,
        user_prompt=user_prompt,
    )

    return {
        "intent": result.get("intent", "general_health"),
        "entities": result.get("entities", []),
        "new_symptoms": result.get("new_symptoms", []),
        "negated_symptoms": result.get("negated_symptoms", []),
        "symptom_details": result.get("symptom_details", {}),
        "body_systems_mentioned": result.get("body_systems_mentioned", []),
    }
