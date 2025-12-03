
from pydantic import BaseModel, Field, ValidationError
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
import re
from utils.azure_llm import getMassGpt

llm = getMassGpt()

class UserQuery(BaseModel):
    query: str = Field(..., min_length=3)
    intent: str = Field(..., description="forecast | reorder | inventory")

MEDICAL_DISALLOWED_PATTERNS = [
    r"\bdiagnose\b", r"\bprescribe\b", r"\bdosage\b", r"\bmedicine\b",
    r"\bdrug\b", r"\btreatment\b", r"\btherapy\b", r"\bwhat disease\b",
    r"\bkill\b", r"\bsuicide\b", r"\bself[- ]?harm\b"
]

def input_guardrails(state: dict) -> dict:
    print("----> validate_input_with_llm")
    query = state.get("query", "").strip()
    if not query:
        return {**state, "error": "EMPTY_QUERY", "output": "Please provide a query."}

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a classifier."),
        ("human", "Classify this query as SAFE or UNSAFE. "
                "Mark UNSAFE if it asks for diagnosis, prescriptions, dosages, or self-harm.\n\n{query}")
        ])
    classification = llm.invoke(prompt.format_prompt(query=query))

    if "UNSAFE" in classification.content.upper():
        return {**state, "error": "DISALLOWED_CONTENT",
                "output": "I cannot provide medical diagnoses, prescriptions, or self-harm guidance. Please consult a licensed healthcare professional."}

    # Regex fallback
    for pat in MEDICAL_DISALLOWED_PATTERNS:
        if re.search(pat, query, flags=re.IGNORECASE):
            return {**state, "error": "DISALLOWED_CONTENT",
                    "output": "Your request involves restricted medical content. Please consult a doctor."}

    return state

def output_guardrails(state: dict) -> dict:
    output = state.get("output", "")

    moderation = llm.invoke({
        "prompt": f"Review the following output. "
                  f"If it contains medical diagnosis, prescriptions, dosages, or unsafe advice, "
                  f"replace it with: 'I cannot provide medical advice. Please consult a licensed healthcare professional.' "
                  f"Otherwise, return the original.\n\n{output}"
    })

    return {**state, "output": moderation}

def route_query_with_llm(state: dict) -> dict:
    print("----> route_query_with_llm")
    query = state["query"]
    print(query)

    # Send the query as a HumanMessage instead of a dict
    intent_msg = llm.invoke([
        HumanMessage(content=f"Determine if this query is about 'appointment' or 'chat'. "
                             f"If it asks for diagnosis, prescriptions, or unsafe medical advice, return 'chat'.\n\n{query}")
    ])

    # Extract the text content from the AIMessage
    intent = intent_msg.content.strip().lower()

    if intent not in {"appointment", "chat"}:
        intent = "chat"

    return intent