from guardrails.guardrail_agent import route_query_with_llm

def route_query(state):
    print("----> Entered Route Query")
    state["intent"] = route_query_with_llm(state)
    print(state["intent"])
    return state