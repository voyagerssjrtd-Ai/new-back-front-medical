from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from utils.azure_llm import getMassGpt
import re
import requests
import xml.etree.ElementTree as ET
import httpx
import json
from bs4 import BeautifulSoup
from datetime import datetime
from utils.embedding import getLargeEmbedding
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

llm = getMassGpt()
embedding_model = getLargeEmbedding()

def getSymptoms(user_query):
    prompt = ChatPromptTemplate.from_messages([
        ("system",
            "You are a medical text extraction assistant. Your task is to identify and extract only the symptoms explicitly mentioned in a patient's query."
        ),
        ("human",
            """Extract the symptoms mentioned in the following query. 
            Rules:
            1. Only return symptoms that are explicitly stated.
            2. Do not infer or add symptoms that are not mentioned.
            3. Return the output as a JSON list under the key 'symptoms'.
            4. Keep the symptom phrases exactly as written by the user.

            Query: {query}
            """
        )
    ])
    formatted_prompt = prompt.format_messages(query=user_query)
    response = llm.invoke(formatted_prompt)
    return response.content

def medCalling(symptom):
    # MedlinePlus Health Topics API URL
    url = "https://wsearch.nlm.nih.gov/ws/query"
    params = {
        "db": "healthTopics",
        "term": symptom
    }
    # Make the request
    response = requests.get(url, params=params)
    if response.status_code == 200:
        # Parse XML response
        root = ET.fromstring(response.text)

        # Extract useful info
        results = []
        for document in root.findall(".//document"):
            title = document.find("content[@name='title']").text if document.find("content[@name='title']") is not None else None
            url = document.find("content[@name='url']").text if document.find("content[@name='url']") is not None else None
            summary = document.find("content[@name='FullSummary']").text if document.find("content[@name='FullSummary']") is not None else None

            results.append({
                "title": title,
                "url": url,
                "summary": summary
            })
        return results
    else:
        raise Exception(f"MedlinePlus API request failed with status code {response.status_code}")
    
def clean_html(raw_html: str) -> str:
    if not raw_html:
        return ""
    soup = BeautifulSoup(raw_html, "html.parser")
    return soup.get_text(separator=" ", strip=True)
    
def build_documents(results, disease_name: str):
    docs = []
    for r in results:
        summary_clean = clean_html(r["summary"])
        title_clean = clean_html(r["title"]) if r["title"] else "untitled"

        docs.append(Document(
            page_content=summary_clean,
            metadata={
                "title": title_clean,
                "url": r["url"] or "unknown",
                "source": "MedlinePlus",
                "disease": disease_name,
                "created_at": datetime.now().isoformat()
            }
        ))
    return docs

def extract_tags(text: str):
    possible_tags = ["Causes", "Symptoms", "Diagnosis", "Treatment", "Prevention", "Complications", "Tests", "Triggers"]
    found_tags = []
    for tag in possible_tags:
        if re.search(rf"\b{tag}\b", text, re.IGNORECASE):
            found_tags.append(tag)
    return found_tags if found_tags else ["general"]

def chunk_documents(txt_documents):
    # Use deterministic chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "]
    )

    chunked_docs = []
    for doc in txt_documents:
        # Split into chunks
        splits = text_splitter.split_text(doc.page_content)

        for idx, chunk in enumerate(splits):
            # Extract tags dynamically from headings/content
            tags = extract_tags(chunk)
            enriched_metadata = {
                "source": doc.metadata.get("source", "unknown"),
                "doc_id": doc.metadata.get("doc_id", f"doc_{idx}"),
                "chunk_id": idx + 1,
                "title": doc.metadata.get("title", "untitled"),
                "tags": ",".join(tags),
                "chunk_length_tokens": len(chunk.split()),  # still approximate, but token-based splitter helps
                "created_at": datetime.now().isoformat()
            }
            chunked_docs.append(Document(page_content=chunk, metadata=enriched_metadata))
    return chunked_docs


def build_vector_store(chunked_docs):
    vector_store = Chroma.from_documents(
            chunked_docs,
            embedding_model,
            persist_directory="chroma_db"
        )
        # Return retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    return retriever

def getMediPlusRetrieverData(user_query):
    llm_response = getSymptoms(user_query)
    parsed = json.loads(llm_response)
    symptoms = parsed.get("symptoms", [])
    print(symptoms)
    symptoms.append("asthma")
    print(symptoms)
    all_docs = []
    for symtom in symptoms:
        results = medCalling(symtom)
        docs = build_documents(results, symtom)
        chunked_docs = chunk_documents(docs)
        all_docs.extend(chunked_docs)
    retriever = build_vector_store(all_docs)
    return retriever