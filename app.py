import streamlit as st

import config
from agent.state import ConversationState, ConversationPhase
from agent.conversation_engine import process_user_turn

st.set_page_config(
    page_title="NeuroHealth - Symptom Assistant",
    page_icon="\U0001f3e5",
    layout="wide",
)

# ── Pre-flight checks ─────────────────────────────────────────────────────────

if not config.GROQ_API_KEY:
    st.error(
        "**GROQ_API_KEY not set.** "
        "Set it in `config.py` or as an environment variable `GROQ_API_KEY`. "
        "Get a free key at [console.groq.com](https://console.groq.com)."
    )
    st.stop()

# ── Session state init ────────────────────────────────────────────────────────

if "conversation_state" not in st.session_state:
    st.session_state.conversation_state = ConversationState()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

state: ConversationState = st.session_state.conversation_state

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("NeuroHealth")
    st.caption("Health Information Assistant")

    st.warning(
        "**Medical Disclaimer**: This tool provides health information only. "
        "It is NOT a substitute for professional medical advice, diagnosis, or treatment. "
        "Always seek the advice of a qualified healthcare provider."
    )

    st.divider()

    # Detected symptoms
    st.subheader("Detected Symptoms")
    if state.confirmed_symptoms:
        for symptom in state.confirmed_symptoms:
            detail = state.symptom_details.get(symptom, {})
            label = symptom.title()
            if detail:
                detail_str = ", ".join(f"{k}: {v}" for k, v in detail.items())
                st.markdown(f"- **{label}** ({detail_str})")
            else:
                st.markdown(f"- {label}")
    else:
        st.caption("No symptoms detected yet.")

    if state.negated_symptoms:
        st.subheader("Ruled Out")
        for s in state.negated_symptoms:
            st.markdown(f"- ~~{s.title()}~~")

    st.divider()

    # Matched conditions
    st.subheader("Matched Conditions")
    if state.top_conditions:
        for cond in state.top_conditions:
            name = cond.get("topic_name") or cond.get("condition", "Unknown")
            url = cond.get("url", "")
            matched = cond.get("matched_symptoms", [])

            if url:
                st.markdown(f"**[{name}]({url})**")
            else:
                st.markdown(f"**{name}**")

            if matched:
                st.caption(f"Matched: {', '.join(matched[:4])}")
    else:
        st.caption("Gathering information...")

    st.divider()

    # Status
    st.subheader("Status")
    phase_labels = {
        ConversationPhase.GREETING: "Ready",
        ConversationPhase.INITIAL_COMPLAINT: "Listening...",
        ConversationPhase.GATHERING_DETAILS: f"Gathering details ({state.followup_round}/{config.MAX_FOLLOWUP_ROUNDS})",
        ConversationPhase.SEARCHING: "Searching medical database...",
        ConversationPhase.PRESENTING_RESULTS: "Results ready",
        ConversationPhase.FOLLOW_UP: "Follow-up",
    }
    st.info(phase_labels.get(state.phase, "Unknown"))

    if st.button("New Conversation", type="primary"):
        st.session_state.conversation_state = ConversationState()
        st.session_state.chat_history = []
        st.rerun()

# ── Main chat area ────────────────────────────────────────────────────────────

st.title("NeuroHealth")
st.caption("Ask about symptoms, conditions, medications, or any health topic.")

# Display chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if user_input := st.chat_input("Ask about symptoms, medications, conditions..."):
    # Display user message
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate and stream assistant response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        try:
            for chunk in process_user_turn(user_input, state):
                full_response += chunk
                response_placeholder.markdown(full_response + "\u258c")

            response_placeholder.markdown(full_response)
        except Exception as e:
            error_msg = f"Something went wrong: {str(e)}"
            response_placeholder.error(error_msg)
            full_response = error_msg

    st.session_state.chat_history.append({"role": "assistant", "content": full_response})
    st.rerun()
