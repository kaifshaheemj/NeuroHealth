from enum import Enum
from dataclasses import dataclass, field


class ConversationPhase(Enum):
    GREETING = "greeting"
    INITIAL_COMPLAINT = "initial_complaint"
    GATHERING_DETAILS = "gathering_details"
    SEARCHING = "searching"
    PRESENTING_RESULTS = "presenting_results"
    FOLLOW_UP = "follow_up"


@dataclass
class ConversationState:
    phase: ConversationPhase = ConversationPhase.GREETING

    # Intent tracking
    query_intent: str = ""
    mentioned_entities: list[str] = field(default_factory=list)
    last_search_query: str = ""

    # Accumulated symptom intelligence
    confirmed_symptoms: list[str] = field(default_factory=list)
    symptom_details: dict = field(default_factory=dict)
    negated_symptoms: list[str] = field(default_factory=list)

    # Conversation tracking
    followup_round: int = 0
    initial_complaint_text: str = ""

    # Search results (cached)
    search_results: list[dict] = field(default_factory=list)
    top_conditions: list[dict] = field(default_factory=list)

    # Conversation history for LLM context
    messages: list[dict] = field(default_factory=list)

    def add_symptoms(self, symptoms: list[str]):
        """Add newly extracted symptoms, dedup against existing."""
        for s in symptoms:
            s_lower = s.lower().strip()
            if s_lower and s_lower not in self.confirmed_symptoms and s_lower not in self.negated_symptoms:
                self.confirmed_symptoms.append(s_lower)

    def add_negated(self, symptoms: list[str]):
        """Add symptoms the user explicitly denied."""
        for s in symptoms:
            s_lower = s.lower().strip()
            if s_lower and s_lower not in self.negated_symptoms:
                self.negated_symptoms.append(s_lower)
                if s_lower in self.confirmed_symptoms:
                    self.confirmed_symptoms.remove(s_lower)

    def add_symptom_detail(self, symptom: str, key: str, value: str):
        """Add a detail about a symptom, e.g. ('headache', 'duration', '3 days')."""
        if symptom not in self.symptom_details:
            self.symptom_details[symptom] = {}
        self.symptom_details[symptom][key] = value

    def should_search(self) -> bool:
        """Decide if we have enough info to search ES."""
        from config import MAX_FOLLOWUP_ROUNDS, MIN_SYMPTOMS_TO_SEARCH
        has_enough = len(self.confirmed_symptoms) >= MIN_SYMPTOMS_TO_SEARCH
        exhausted = self.followup_round >= MAX_FOLLOWUP_ROUNDS
        return has_enough or exhausted

    def add_entities(self, entities: list[str]):
        """Add mentioned entities (drug names, condition names, etc.), dedup."""
        for e in entities:
            e_lower = e.lower().strip()
            if e_lower and e_lower not in self.mentioned_entities:
                self.mentioned_entities.append(e_lower)

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
