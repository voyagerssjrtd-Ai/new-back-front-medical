import re
from azure_llm import getMassGpt
from database.loader import load_sales_history
from chunking import getRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# ----------------------------------------------------
# Initialize LLM
# ----------------------------------------------------
llm = getMassGpt()


# ====================================================
# 1. FORECAST HORIZON PARSER
# ====================================================
def parse_horizon(user_query: str) -> int:
    import re

    text = str(user_query).lower()
    default = 6 

    # Map text numbers to integers
    numbers = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "eleven": 11, "twelve": 12
    }

    # Numeric months
    m = re.search(r"(?:next\s+)?(\d+)\s+months?", text)
    if m:
        return int(m.group(1))

    # Numeric years
    y = re.search(r"(?:next\s+)?(\d+)\s+years?", text)
    if y:
        return int(y.group(1)) * 12

    # Text months
    tm = re.search(r"(?:next\s+)?(\w+)\s+months?", text)
    if tm and tm.group(1) in numbers:
        return numbers[tm.group(1)]

    # Text years
    ty = re.search(r"(?:next\s+)?(\w+)\s+years?", text)
    if ty and ty.group(1) in numbers:
        return numbers[ty.group(1)] * 12

    # Default fallback
    return default




# ====================================================
# 2. LLM-GENERATED RAG QUERY
# ====================================================
def generate_retrieval_query(product: str, horizon: int, user_query: str) -> str:
    prompt = f"""
You are a retrieval query generator for sales forecasting.

Rewrite the user's message into a *high-recall* vector DB search query.

PRODUCT: {product}
FORECAST HORIZON: {horizon} months
USER QUERY: {user_query}

Your retrieval query MUST target:
- FMCG rural vs urban consumption trends
- GST disruptions
- Category-level trends (HPC, food, staples, impulse)
- Volume/value growth signals
- Smaller pack affordability shifts
- E-commerce vs modern trade changes
- 2026 retail marketing calendar seasonal demand triggers
- Macro disruptions affecting consumption

Return ONLY the rewritten search query text.
"""

    return llm.invoke(prompt).content.strip()


# ====================================================
# 3. RAG CHAIN (Retrieval → Ranking LLM)
# ====================================================
def build_rag_chain():
    system_message = """
You are a market insights extraction agent.

Given retrieved documents, return ONLY the most relevant parts for forecasting.

Return list elements in the exact literal format:

{{
  "source_id": "...",
  "chunk_text": "...",
  "chunk_date": "...",
  "relevance_score": "..."
}}

Focus on:
- FMCG sector growth
- rural/urban pattern shifts
- GST-related disruptions
- seasonal marketing calendar effects
- category trends (HPC, food, staples)
- affordability/smaller pack patterns
- e-commerce vs modern trade shifts
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", "CONTEXT:\n{context}\n\nSEARCH QUERY: {question}")
    ])

    rag_chain = (
        {
            "context": lambda inp: "\n\n".join([d.page_content for d in inp["docs"]]),
            "question": lambda inp: inp["query"]
        }
        | prompt
        | llm
    )

    return rag_chain


# ====================================================
# 4. FINAL FORECASTING AGENT
# ====================================================
def forecasting_agent(state: dict):

    sku = state["sku"]
    user_query = state["query"]

    # SKU → product name mapping
    sku_map = {
        "SKU101": "Laptop",
        "SKU102": "Chair",
        "SKU103": "Printer Ink",
        "SKU104": "Desk Lamp",
        "SKU105": "USB Cable"
    }
    product = sku_map.get(sku, "Unknown Product")

    # Load sales
    sales = load_sales_history()
    if sku:
        sales = sales[sales["item_id"] == sku]

    # ====================================================
    # Forecast horizon (fixed to correctly handle years)
    # ====================================================
    horizon = parse_horizon(user_query)

    # Generate optimal retrieval query
    retrieval_query = generate_retrieval_query(product, horizon, user_query)

    # Vector retrieval
    retriever = getRetriever()
    docs = retriever.invoke(retrieval_query)

    # Build and run RAG ranking chain
    rag_chain = build_rag_chain()
    rag_output = rag_chain.invoke({"docs": docs, "query": retrieval_query})

    # ==================================================
    #  FINAL FORECAST PROMPT (ALL BRACES ESCAPED)
    # ==================================================
    forecast_prompt = f"""
You are an enterprise forecasting engine.

Use the provided:
1. Historical sales data:
{sales}

2. Retrieved insights:
{rag_output}

3. Product: {product}
4. Forecast horizon: {horizon} months
   - Generate exactly {horizon} months of forecast data, not just 12 months.
   - Include baseline, upside, and downside for each month.

TASKS:
- Validate and preprocess sales data
- Detect monthly trend, seasonality, missing periods
- Map extracted insights into numerical effects

Example of the effect mapping output (escape braces handled):
{{
  "source_id": "...",
  "claim": "...",
  "mapped_effect": "+3% demand lift expected in rural markets",
  "confidence": 0.82
}}

Build:
- Baseline scenario
- Upside scenario
- Downside scenario
- Seasonal adjustments using 2026 calendar info
- Adjustments from FMCG rural/urban patterns and GST impact

OUTPUT JSON ONLY (escape braces preserved):
{{
  "sku": "{sku}",
  "forecast_horizon_months": {horizon},
  "model_used": "llm_rag_hybrid_forecaster",
  "data_summary": {{}} ,
  "preprocessing_log": [],
  "insight_summary": [],
  "forecast_table": [],
  "scenarios": {{}} ,
  "explainability": {{}} ,
  "recommendations": [],
  "error": null
}}

EXECUTIVE SUMMARY:
- One sentence with forecast numbers
- Three actionable recommendations referencing source_id
"""


    final_output = llm.invoke(forecast_prompt)

    # Save results in state
    state["retrieval_query"] = retrieval_query
    state["retrieved_chunks"] = rag_output
    state["output"] = final_output.content

    return state

#---------------------------------------------------------------------
# >>>>>>>>>>>>>>>>>>> UI <<<<<<<<<<<<<<<<<<
#-----------------------------------------------------------------------

# import json

# def format_for_ui(raw_output: dict) -> dict:
#     return {
#         "metadata": {
#             "sku": raw_output.get("sku"),
#             "forecast_horizon_months": raw_output.get("forecast_horizon_months"),
#             "model_used": raw_output.get("model_used")
#         },
#         "historical_data": raw_output.get("data_summary", {}),
#         "preprocessing": raw_output.get("preprocessing_log", []),
#         "insights": [
#             {
#                 "source_id": i.get("source_id"),
#                 "claim": i.get("claim"),
#                 "effect": i.get("mapped_effect"),
#                 "confidence": i.get("confidence")
#             } for i in raw_output.get("insight_summary", [])
#         ],
#         "forecast": [
#             {
#                 "month": f.get("month"),
#                 "baseline": f.get("baseline_quantity"),
#                 "upside": f.get("upside_quantity"),
#                 "downside": f.get("downside_quantity")
#             } for f in raw_output.get("forecast_table", [])
#         ],
#         "scenarios": raw_output.get("scenarios", {}),
#         "explainability": raw_output.get("explainability", {}),
#         "recommendations": raw_output.get("recommendations", []),
#         "executive_summary": raw_output.get("executive_summary", ""),
#         "error": raw_output.get("error", None)
#     }

