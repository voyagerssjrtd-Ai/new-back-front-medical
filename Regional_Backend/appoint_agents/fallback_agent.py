def fallback_appoint_agent(state: dict) -> dict:
    state["status"] = "fallback"
    state["output"] = "Sorry, I couldn’t classify your request into doctor, lab, disease, or service."
    return state