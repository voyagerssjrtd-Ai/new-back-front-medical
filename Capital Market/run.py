from workflow import app
import json

user_input = input("Enter request: ")  # e.g., "forecast SKU001"

# Smart SKU detection
sku = None
for word in user_input.split():
    if word.upper().startswith("SKU"):
        sku = word.upper()

result = app.invoke({
    "query": user_input,
    "sku": sku
})

print("\n====== SMART INVENTORY AGENT OUTPUT ======")
print(result["intent"])
print(result["output"])
print("==========================================\n")