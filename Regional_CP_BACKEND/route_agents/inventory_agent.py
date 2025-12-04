from database.loader import load_inventory
from utils.azure_llm import getMassGpt

llm = getMassGpt()

def inventory_agent(state):
    sku = state.get("sku")
    inventory = load_inventory()

    if sku not in inventory:
        state["output"] = "SKU not found."
        return state

    data = inventory[sku]

    prompt = f"""
Provide an enterprise-grade inventory status summary.

Item:
SKU: {sku}
Name: {data['name']}
Stock: {data['stock']}
Threshold: {data['threshold']}
Tasks:
- Stock health
- Risk of stockout
- Next action
- Business notes
"""

    resp = llm.invoke(prompt)
    state["output"] = resp.content
    return state
