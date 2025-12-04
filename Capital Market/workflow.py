from langgraph.graph import StateGraph, END

from typing_extensions import TypedDict
from routing_agent import route_query
from forecast_agent import forecasting_agent
from reorder_agent import reorder_agent
from inventory_agent import inventory_agent

class State(TypedDict):
    sku : str
    query : str
    intent : str
    output : str

graph = StateGraph(State)

graph.set_entry_point("route_query")

graph.add_node("route_query", route_query)
graph.add_node("forecast", forecasting_agent)
graph.add_node("reorder", reorder_agent)
graph.add_node("inventory", inventory_agent)

graph.add_conditional_edges(
    "route_query",
    lambda s: s["intent"],
    {
        "forecast": "forecast",
        "reorder": "reorder",
        "inventory": "inventory"
    }
)

graph.add_edge("forecast", END)
graph.add_edge("reorder", END)
graph.add_edge("inventory", END)

app = graph.compile()
