SYSTEM_PROMPT_BASE = """You are NeuroHealth, a friendly and knowledgeable health information assistant.
You help users understand their symptoms and find relevant health information.

CRITICAL RULES:
- You are NOT a doctor. You do NOT diagnose conditions.
- Always include a disclaimer that users should consult a healthcare professional.
- If symptoms suggest an emergency (chest pain + shortness of breath, sudden severe headache,
  signs of stroke like facial drooping or slurred speech, difficulty breathing), immediately
  advise calling emergency services.
- Be empathetic, clear, and avoid medical jargon unless explaining it.
- Ask focused follow-up questions to better understand symptoms.
- Never suggest specific medications or dosages.
- When presenting possible conditions, phrase them as "conditions commonly associated with
  these symptoms" not as diagnoses.

CONVERSATION APPROACH:
1. Listen to the user's initial complaint carefully.
2. Ask 2-3 focused follow-up questions to gather: duration, severity, location,
   associated symptoms, triggers, and relevant history.
3. After gathering enough information, present findings from the medical knowledge base.
4. Offer to explain any condition in more detail or explore related topics.
"""

FOLLOWUP_QUESTION_PROMPT = """Based on the user's reported symptoms and conversation so far,
ask ONE focused follow-up question to gather more diagnostic information.

Current confirmed symptoms: {symptoms}
Known details: {details}
Follow-up round: {round} of {max_rounds}

Guidelines for your question:
- Round 1: Ask about DURATION and SEVERITY of the primary symptom.
- Round 2: Ask about ASSOCIATED symptoms (suggest 2-3 specific ones that commonly co-occur).
- Round 3: Ask about TRIGGERS, recent changes, or medical history relevant to these symptoms.

Keep your question concise, empathetic, and natural. Do not list multiple questions at once.
Ask only ONE question."""

RAG_CONTEXT_TEMPLATE = """
--- MEDICAL KNOWLEDGE BASE RESULTS ---
The following information was retrieved from the MedlinePlus medical database.
Use this to inform your response, but present it naturally (do not say "according to the database").

{search_results}

--- END OF KNOWLEDGE BASE ---

Based on the user's symptoms ({symptom_list}) and the retrieved medical knowledge above,
provide a helpful, empathetic response that:
1. Acknowledges what the user is experiencing.
2. Lists 2-4 conditions commonly associated with these symptoms, with brief explanations.
3. For each condition, mention key distinguishing features.
4. Recommend appropriate next steps (e.g., see a doctor, urgent care, or emergency).
5. End with a disclaimer: "This information is for educational purposes only and is not a
   substitute for professional medical advice. Please consult a healthcare provider for
   proper evaluation."
"""


def build_rag_context_string(search_results: list[dict]) -> str:
    """Format ES search results into a string for LLM context injection."""
    if not search_results:
        return "No specific conditions found in the knowledge base for these symptoms."

    lines = []
    for i, r in enumerate(search_results, 1):
        topic = r.get("topic_name") or r.get("condition", "Unknown")
        topic_type = r.get("topic_type", "")
        summary = r.get("summary", "")[:300]
        symptoms = r.get("symptoms", [])
        url = r.get("url", "")
        body_systems = r.get("body_systems", [])
        matched = r.get("matched_symptoms", [])

        entry = f"[{i}] {topic}"
        if topic_type:
            entry += f" ({topic_type})"
        if body_systems:
            entry += f" | Systems: {', '.join(body_systems)}"
        if matched:
            entry += f" | Matched symptoms: {', '.join(matched[:5])}"
        if symptoms:
            entry += f"\n    Known symptoms: {', '.join(symptoms[:8])}"
        if summary:
            entry += f"\n    Summary: {summary}"
        if url:
            entry += f"\n    Source: {url}"
        lines.append(entry)

    return "\n\n".join(lines)


def build_system_prompt_with_rag(state, search_results: list[dict]) -> str:
    """Build the full system prompt with RAG context injected."""
    rag_text = build_rag_context_string(search_results)
    symptom_list = ", ".join(state.confirmed_symptoms)

    rag_section = RAG_CONTEXT_TEMPLATE.format(
        search_results=rag_text,
        symptom_list=symptom_list,
    )

    return SYSTEM_PROMPT_BASE + "\n\n" + rag_section


# ── Intent-specific RAG templates ────────────────────────────────────────────

DRUG_RAG_CONTEXT_TEMPLATE = """
--- MEDICATION KNOWLEDGE BASE RESULTS ---
The following drug/medication information was retrieved from the MedlinePlus medical database.

{search_results}

--- END OF KNOWLEDGE BASE ---

The user is asking about: {entity_list}

Provide a helpful response that covers:
1. What the medication is and what it is commonly used for.
2. Common side effects and serious side effects to watch for.
3. Important safety information (age restrictions, pregnancy, interactions with other drugs).
4. General dosage guidance (e.g., "typically taken every 6-8 hours" — never give exact dosages).
5. When to seek medical attention related to this medication.

CRITICAL RULES:
- Do NOT prescribe or recommend specific dosages. Say "consult your pharmacist or doctor for proper dosing."
- If asked about children's safety, mention that pediatric dosing differs and a pediatrician should be consulted.
- If the knowledge base does not contain the specific drug requested, say so honestly and provide what general information is available from related drug categories.
- End with: "This information is for educational purposes only. Always consult a healthcare provider or pharmacist before starting, stopping, or changing any medication."
"""

CONDITION_INFO_RAG_CONTEXT_TEMPLATE = """
--- CONDITION KNOWLEDGE BASE RESULTS ---
The following condition/disease information was retrieved from the MedlinePlus medical database.

{search_results}

--- END OF KNOWLEDGE BASE ---

The user is asking about: {entity_list}

Provide a helpful response that covers:
1. What the condition is (clear, plain-language explanation).
2. Common symptoms and how the condition typically presents.
3. Known causes or risk factors.
4. Common treatment approaches (general categories, not specific prescriptions).
5. When to see a doctor about this condition.
6. If relevant, mention related conditions the user might want to know about.

CRITICAL RULES:
- Do NOT diagnose the user. Present this as general information.
- If asked "do I have this?", redirect them to consult a healthcare provider.
- End with: "This information is for educational purposes only and is not a substitute for professional medical advice."
"""

GENERAL_HEALTH_RAG_CONTEXT_TEMPLATE = """
--- HEALTH KNOWLEDGE BASE RESULTS ---
The following health information was retrieved from the MedlinePlus medical database.

{search_results}

--- END OF KNOWLEDGE BASE ---

The user asked about: {entity_list}

Provide a helpful, evidence-based response using the retrieved information.
Be practical and actionable. If the knowledge base has relevant information, use it.
If the topic is not well covered, provide what general guidance is available and suggest consulting a healthcare provider.

End with: "This information is for educational purposes only. Consult a healthcare provider for personalized advice."
"""


def build_drug_system_prompt(state, search_results: list[dict]) -> str:
    """Build system prompt for drug/medication queries."""
    rag_text = build_rag_context_string(search_results)
    entity_list = ", ".join(state.mentioned_entities) if state.mentioned_entities else "the medication"

    rag_section = DRUG_RAG_CONTEXT_TEMPLATE.format(
        search_results=rag_text,
        entity_list=entity_list,
    )
    return SYSTEM_PROMPT_BASE + "\n\n" + rag_section


def build_condition_system_prompt(state, search_results: list[dict]) -> str:
    """Build system prompt for condition information queries."""
    rag_text = build_rag_context_string(search_results)
    entity_list = ", ".join(state.mentioned_entities) if state.mentioned_entities else "the condition"

    rag_section = CONDITION_INFO_RAG_CONTEXT_TEMPLATE.format(
        search_results=rag_text,
        entity_list=entity_list,
    )
    return SYSTEM_PROMPT_BASE + "\n\n" + rag_section


def build_general_health_system_prompt(state, search_results: list[dict]) -> str:
    """Build system prompt for general health queries."""
    rag_text = build_rag_context_string(search_results)
    entity_list = ", ".join(state.mentioned_entities) if state.mentioned_entities else "the topic"

    rag_section = GENERAL_HEALTH_RAG_CONTEXT_TEMPLATE.format(
        search_results=rag_text,
        entity_list=entity_list,
    )
    return SYSTEM_PROMPT_BASE + "\n\n" + rag_section


INTENT_PROMPT_BUILDERS = {
    "drug_query": build_drug_system_prompt,
    "condition_info": build_condition_system_prompt,
    "general_health": build_general_health_system_prompt,
    "symptom_report": build_system_prompt_with_rag,
}
