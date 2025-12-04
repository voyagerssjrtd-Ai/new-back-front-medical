from utils.guardrails import validate_input, parse_intent

def route_query(state):
    user_input = state["query"]
    validate_input(user_input)
    state["intent"] = parse_intent(user_input)
    return state