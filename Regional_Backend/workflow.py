# from langgraph.graph import StateGraph, END
# from typing_extensions import TypedDict
# from routing_agent import route_query
# from route_agents.appointment_agent import appointment_book_agent
# from route_agents.chat_agent import chat_query_agent

# class State(TypedDict):
#     query : str
#     intent : str
#     kind: str
#     doctor_name: str
#     specialty: str
#     department: str
#     test: str
#     disease: str
#     date: str
#     time: str
#     location: str
#     preferred_start: str
#     status : str
#     booked_slot : str
#     alternatives : str
#     output : str

# graph = StateGraph(State)
# print("----> Entered Work Flow")

# graph.set_entry_point("route_query")

# graph.add_node("route_query", route_query)
# graph.add_node("appointment", appointment_book_agent)
# graph.add_node("chat", chat_query_agent)

# graph.add_conditional_edges(
#     "route_query",
#     lambda s: s["intent"],
#     {
#         "appointment": "appointment",
#         "chat": "chat",
#     }
# )

# graph.add_edge("appointment", END)
# graph.add_edge("chat", END)

# app = graph.compile()

from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict
from routing_agent import route_query
from route_agents.appointment_agent import appointment_book_agent
from route_agents.chat_agent import chat_query_agent
from guardrails.guardrail_agent import input_guardrails

class State(TypedDict):
    query : str
    intent : str
    kind: str
    doctor_name: str
    specialty: str
    department: str
    test: str
    disease: str
    date: str
    time: str
    location: str
    preferred_start: str
    status : str
    booked_slot : str
    alternatives : str
    output : str

graph = StateGraph(State)
print("----> Entered Work Flow")

graph.set_entry_point("input_guardrails")

graph.add_node("input_guardrails", input_guardrails)
graph.add_node("route_query", route_query)
graph.add_node("appointment", appointment_book_agent)
graph.add_node("chat", chat_query_agent)

graph.add_edge("input_guardrails","route_query")
graph.add_conditional_edges(
    "route_query",
    lambda s: s["intent"],
    {
        "appointment": "appointment",
        "chat": "chat",
    }
)
graph.add_edge("appointment", END)
graph.add_edge("chat", END)

app = graph.compile()