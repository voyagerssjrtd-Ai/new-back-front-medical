import json
from cm_ingestion_agent import ingestion_agent
from database.loader import load_jsons

# Step 2: Prepare state and call ingestion agent
def main():
    # Load transactions
    transactions = load_jsons()
    print(f"Loaded {len(transactions)} transactions from JSON file.\n")

    # Create initial state
    state = {"transactions": transactions}

    # Run Data Ingestion & Preprocessing Agent
    processed_state = ingestion_agent(state)

    # Output results
    print("\n--- Clean Transactions ---")
    for t in processed_state["clean_transactions"]:
        print(t)
    
    print("\n--- Data Anomalies ---")
    for a in processed_state["data_anomalies"]:
        print(a)

if __name__ == "__main__":
    main()
