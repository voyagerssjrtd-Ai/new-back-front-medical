import datetime
from cm_logger import log

MANDATORY_FIELDS = [
    "trade_id","instrument","isin","trade_date","settlement_date","buyer_lei","seller_lei",
    "price","quantity","trade_type","venue"]

# Validation functions
def is_valid_date(date_str):
    try:
        datetime.datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except ValueError:
        return False

def is_valid_isin(isin):
    return isinstance(isin, str) and len(isin) == 12 and isin.isalnum()

def is_valid_lei(lei):
    return isinstance(lei, str) and len(lei) == 20 and lei.isalnum()

def ingestion_agent(state):

    log("Running Data Ingestion & Preprocessing Agent...")

    transactions = state.get("transactions", [])
    cleaned = []
    anomalies = []
    for t in transactions:
        trade_anomalies = []

        # 1 Check Mandatory Fields
        for field in MANDATORY_FIELDS:
            if field not in t or t[field] in [None, "", 0]:
                trade_anomalies.append(f"Missing mandatory field: {field}")

        # 2️ ISIN validation
        if "isin" in t and not is_valid_isin(t["isin"]):
            trade_anomalies.append("Invalid ISIN format")

        # 3️ LEI validation
        if "buyer_lei" in t and not is_valid_lei(t["buyer_lei"]):
            trade_anomalies.append("Invalid Buyer LEI format")
        if "seller_lei" in t and not is_valid_lei(t["seller_lei"]):
            trade_anomalies.append("Invalid Seller LEI format")

        # 4️ Date validation
        if "trade_date" in t and not is_valid_date(t["trade_date"]):
            trade_anomalies.append("Invalid trade_date format")
        if "settlement_date" in t and not is_valid_date(t["settlement_date"]):
            trade_anomalies.append("Invalid settlement_date format")

        # 5️ Numeric Checks
        if "price" in t and (not isinstance(t["price"], (int, float)) or t["price"] <= 0):
            trade_anomalies.append("Price must be a positive number")
        if "quantity" in t and (not isinstance(t["quantity"], int) or t["quantity"] <= 0):
            trade_anomalies.append("Quantity must be a positive integer")

        # Add to cleaned list
        cleaned.append(t)

        # Record anomalies
        if trade_anomalies:
            anomalies.append({
                "trade_id": t.get("trade_id", "UNKNOWN"),
                "issues": trade_anomalies
            })

    # Save results back to state
    state["clean_transactions"] = cleaned
    state["data_anomalies"] = anomalies

    log(f" Ingestion complete: {len(cleaned)} transactions processed.")
    log(f"Found {len(anomalies)} anomalies.")

    return state
