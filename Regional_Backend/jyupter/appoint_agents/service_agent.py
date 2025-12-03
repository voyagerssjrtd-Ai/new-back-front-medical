from app.common_util import find_matching_slots,book_slot,suggest_alternatives_json

def service_appoint_agent(state: dict) -> dict:
    if not state.get("service"):
        state["status"] = "error"
        state["output"] = "Service name missing. Please specify the service."
        return state

    preferred, alternatives = find_matching_slots("service", state["service"], state["preferred_start"])
    if preferred and book_slot("service", state["service"], state["preferred_start"]):
        state["status"] = "confirmed"
        state["booked_slot"] = state["preferred_start"]
        state["output"] = f"Service '{state['service']}' booked for {state['preferred_start']}."
    else:
        state["status"] = "unavailable"
        state["alternatives"] = suggest_alternatives_json(alternatives)
        state["output"] = "Requested service slot unavailable. Alternatives provided."
    return state