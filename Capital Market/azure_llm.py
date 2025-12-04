from langchain_openai import ChatOpenAI
import httpx

def getMassGpt():
    # Create an HTTP client that skips SSL verification (only for hackathon/test environments)
    client = httpx.Client(verify=False)
    llm = ChatOpenAI(
        base_url="https://genailab.tcs.in",
        model="azure/genailab-maas-gpt-4o",
        api_key="sk-u6zTQaiDKlhHn4-k_hhihw",
        http_client=client
    )
    return llm
