from app.common_util import find_matching_slots,book_slot,suggest_alternatives_json
def doctor_appoint_agent(state: dict) -> dict:
    print("Entered doctor_appoint_agent")

    doctor_name = state.get("doctor_name")
    if not doctor_name:
        state["status"] = "error"
        state["output"] = "Doctor name missing. Please provide doctor details."
        return state

    # Lookup doctor_id from doctors table using doctor_name
    import sqlite3
    with sqlite3.connect("triage.db") as conn:
        c = conn.cursor()
        c.execute("SELECT id FROM doctors WHERE name=?", (doctor_name,))
        row = c.fetchone()
        if not row:
            state["status"] = "error"
            state["output"] = f"No doctor found with name '{doctor_name}'."
            return state
        doctor_id = row[0]

    # Now use doctor_id for slot matching and booking
    preferred, alternatives = find_matching_slots("doctor", doctor_id, state["preferred_start"])
    if preferred and book_slot("doctor", doctor_id, state["preferred_start"]):
        state["status"] = "confirmed"
        state["booked_slot"] = state["preferred_start"]
        state["output"] = f"Doctor appointment with {doctor_name} confirmed for {state['preferred_start']}."
        state["alternatives"] = "[]"  # always include alternatives key
    else:
        state["status"] = "unavailable"
        state["alternatives"] = suggest_alternatives_json(alternatives)
        state["output"] = f"Requested slot with {doctor_name} unavailable. Alternatives provided."

    return state