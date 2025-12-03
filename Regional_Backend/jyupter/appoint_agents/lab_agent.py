from app.common_util import find_matching_slots,book_slot,suggest_alternatives_json

def lab_appoint_agent(state: dict) -> dict:
    if not state.get("lab_id"):
        state["status"] = "error"
        state["output"] = "Lab ID missing. Please provide lab details."
        return state

    preferred, alternatives = find_matching_slots("lab", state["lab_id"], state["preferred_start"])
    if preferred and book_slot("lab", state["lab_id"], state["preferred_start"]):
        state["status"] = "confirmed"
        state["booked_slot"] = state["preferred_start"]
        state["output"] = f"Lab test '{state.get('test','unspecified')}' booked for {state['preferred_start']} at {state.get('location','default lab')}."
    else:
        state["status"] = "unavailable"
        state["alternatives"] = suggest_alternatives_json(alternatives)
        state["output"] = "Requested lab slot unavailable. Alternatives provided."
    return state