from utils.common_util import find_matching_slots,book_slot,suggest_alternatives_json

def disease_appoint_agent(state: dict) -> dict:
    if not state.get("disease"):
        state["status"] = "error"
        state["output"] = "Disease information missing. Please specify the disease."
        return state

    preferred, alternatives = find_matching_slots("disease", state["disease"], state["preferred_start"])
    if preferred and book_slot("disease", state["disease"], state["preferred_start"]):
        state["status"] = "confirmed"
        state["booked_slot"] = state["preferred_start"]
        state["output"] = f"Consultation for {state['disease']} confirmed for {state['preferred_start']}."
    else:
        state["status"] = "unavailable"
        state["alternatives"] = suggest_alternatives_json(alternatives)
        state["output"] = "Requested consultation slot unavailable. Alternatives provided."
    return state