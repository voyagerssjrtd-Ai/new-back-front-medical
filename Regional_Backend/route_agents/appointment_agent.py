import json
import re
from datetime import datetime, timedelta
from langchain_core.messages import HumanMessage
from utils.azure_llm import getMassGpt

from appoint_agents.doctor_agent import doctor_appoint_agent
from appoint_agents.lab_agent import lab_appoint_agent
from appoint_agents.disease_agent import disease_appoint_agent
from appoint_agents.service_agent import service_appoint_agent
from appoint_agents.fallback_agent import fallback_appoint_agent

llm = getMassGpt()

# --- Helpers ---

def extract_entities(query: str) -> dict:
    """Call LLM to extract structured entities from the query."""
    raw_entities_msg = llm.invoke([
        HumanMessage(content=f"""
        Extract entities from this appointment request.
        Return ONLY valid JSON, no extra text, no explanations.
        Keys: type (doctor/lab/disease/service), doctor_name, specialty, department,
              test, disease, service, date, time, location.
        Rules:
        - If no date is mentioned, set "date" to "tomorrow".
        - If no time is mentioned, set "time" to "09:00".
        Input: "{query}"
        """)
    ])
    raw_text = raw_entities_msg.content.strip()
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if match:
        raw_text = match.group(0)

    try:
        return json.loads(raw_text)
    except Exception:
        print("Failed to parse JSON from LLM output:", raw_text)
        return {}

def normalize_date(date_str: str) -> str:
    """Normalize date string to YYYY-MM-DD, default to tomorrow."""
    if not date_str or date_str.lower() == "tomorrow":
        return (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").strftime("%Y-%m-%d")
    except Exception:
        return (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

def normalize_time(time_str: str) -> str:
    """Normalize time string, default to 09:00."""
    return time_str or "09:00"

# --- Main Agent ---

def appointment_book_agent(state: dict):
    query = state.get("query", "")
    print("Entered Appointment Booking")

    entities = extract_entities(query)

    parsed_date = normalize_date(entities.get("date"))
    time_str = normalize_time(entities.get("time"))

    # Update state with extracted values
    state.update({
        "kind": entities.get("type"),
        "doctor_name": entities.get("doctor_name"),
        "specialty": entities.get("specialty"),
        "department": entities.get("department"),
        "test": entities.get("test"),
        "disease": entities.get("disease"),
        "service": entities.get("service"),
        "date": parsed_date,
        "time": time_str,
        "location": entities.get("location"),
        "preferred_start": f"{parsed_date} {time_str}"
    })

    print("Routing based on kind:", state["kind"])

    # Routing map
    agent_map = {
        "doctor": doctor_appoint_agent,
        "lab": lab_appoint_agent,
        "disease": doctor_appoint_agent,  # disease handled by doctor agent
        "service": service_appoint_agent,
    }

    agent = agent_map.get(state["kind"], fallback_appoint_agent)
    return agent(state)