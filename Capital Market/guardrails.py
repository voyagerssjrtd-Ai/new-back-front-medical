from pydantic import BaseModel, Field, ValidationError

class UserQuery(BaseModel):
    query: str = Field(..., min_length=3)
    intent: str = Field(..., description="forecast | reorder | inventory")

FORBIDDEN = ["hack", "illegal", "explode", "virus"]

def validate_input(text: str):
    for bad in FORBIDDEN:
        if bad in text.lower():
            raise Exception("❌ Forbidden content detected.")
    return True


def parse_intent(user_query: str):
    q = user_query.lower()
    if "forecast" in q or "predict" in q:
        return "forecast"
    if "reorder" in q or "replenish" in q:
        return "reorder"
    if "stock" in q or "inventory" in q:
        return "inventory"
    return "inventory"