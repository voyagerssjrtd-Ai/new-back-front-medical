from utils.azure_llm import getMassGpt
from database.loader import load_inventory

llm = getMassGpt()

def reorder_agent(state):
    sku = state.get("sku")
    inventory = load_inventory()

    if sku not in inventory:
        state["output"] = "SKU not found."
        return state

    item = inventory[sku]

    prompt = f"""
You are an enterprise supply chain replenishment agent.

Item:
SKU: {sku}
Name: {item['name']}
Current Stock: {item['stock']}
Threshold: {item['threshold']}

Tasks:
- Should we reorder?
- How much?
- Provide business justification.
- Add risk score.

Return JSON result.
"""

    resp = llm.invoke(prompt)
    state["output"] = resp.content
    return state

# -------------------------------------------------------------------
# Reflection Agent 
# -------------------------------------------------------------------

# from azure_llm import getMassGpt
# from database.loader import load_inventory

# llm = getMassGpt()

# def reorder_agent_with_reflection(state, max_iterations=2):
#     sku = state.get("sku")
#     inventory = load_inventory()

#     if sku not in inventory:
#         state["output"] = "SKU not found."
#         return state

#     item = inventory[sku]

#     # Initial prompt
#     prompt_template = f"""
# You are an enterprise supply chain replenishment agent.

# Item:
# SKU: {sku}
# Name: {item['name']}
# Current Stock: {item['stock']}
# Threshold: {item['threshold']}

# Tasks:
# - Should we reorder?
# - How much?
# - Provide business justification.
# - Add risk score.

# Return JSON result.
# """
#     iteration = 0
#     final_output = None

#     while iteration < max_iterations:
#         iteration += 1

#         # Step 1: Agent generates output
#         resp = llm.invoke(prompt_template)
#         output_json = resp.content.strip()

#         # Step 2: Reflection - check if output is valid JSON and makes sense
#         reflection_prompt = f"""
# You are a reflection agent.

# Check the following JSON output from a reorder agent:
# {output_json}

# Tasks:
# 1. Is it valid JSON? If not, correct it.
# 2. Does it contain: reorder recommendation, quantity, justification, and risk score? If not, add missing fields.
# 3. Return corrected JSON ONLY.
# """

#         reflection_resp = llm.invoke(reflection_prompt)
#         corrected_output = reflection_resp.content.strip()

#         # Optional: simple check for improvement
#         if corrected_output == output_json:
#             # No changes needed
#             final_output = corrected_output
#             break
#         else:
#             # Update output and try one more iteration
#             output_json = corrected_output
#             final_output = corrected_output

#     state["output"] = final_output
#     return state

