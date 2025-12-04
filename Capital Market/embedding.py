from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import httpx

client = httpx.Client(verify=False)
def getLargeEmbedding():
    embedding_model = OpenAIEmbeddings(
        base_url="https://genailab.tcs.in",
        model="azure/genailab-maas-text-embedding-3-large",
        api_key="sk-u6zTQaiDKlhHn4-k_hhihw",
        http_client=client
    )
    return embedding_model

