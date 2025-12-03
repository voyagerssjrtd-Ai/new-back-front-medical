from langchain_core.prompts import ChatPromptTemplate
from data.loader import load_txts
from utils.azure_llm import getMassGpt
from utils.embedding import getLargeEmbedding
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
import re
from datetime import datetime

# ----------------------------------------------------------
#         FIRST WAY
# ----------------------------------------------------------

# llm = getMassGpt()
# embedding_model = getLargeEmbedding()

# def getRetriever():
#     txt_documents = load_txts()
#     chunked_docs = []
#     for doc in txt_documents:
#         prompt = ChatPromptTemplate.from_messages([
#             ("system", 
#                 "You are a helpful assistant that splits medical or disease-related documents "
#                 "into semantically meaningful chunks for downstream semantic search and retrieval."
#             ),
#             ("human", 
#                 """You will be given a document. Split it into semantically coherent sections, following these rules:

#                 1. Do not omit any information.
#                 2. Keep related items together.
#                 3. Each chunk should be self-contained.
#                 4. Aim for ~5000–6000 words per chunk.
#                 5. Return chunks as a numbered list with headings like '### Chunk 1:', '### Chunk 2:'.
#                 6. Include metadata (e.g., source) at the beginning of each chunk.

#                 Document:
#                 {document}
#                 """
#             )
#         ])

#         formatted_prompt = prompt.format_messages(document=doc.page_content)
#         response = llm.invoke(formatted_prompt)

#         chunks = [c.strip() for c in response.content.split("### Chunk") if c.strip()]
#         for idx, chunk in enumerate(chunks):
#             chunked_docs.append(
#                 Document(
#                     page_content=chunk,
#                     metadata={"source": f"{doc.metadata['source']}_chunk{idx+1}"}
#                 )
#             )
#     vector_store = Chroma.from_documents(
#         chunked_docs,
#         embedding_model,
#         persist_directory="chroma_db"
#     )

#     retriever = vector_store.as_retriever(search_kwargs={"k": 3})
#     return retriever


# ----------------------------------------------------------
#         SECOND WAY
# ----------------------------------------------------------

# def extract_tags(text: str):
#     possible_tags = ["Causes", "Symptoms", "Diagnosis", "Treatment", "Prevention", "Complications", "Tests", "Triggers"]
#     found_tags = []
#     for tag in possible_tags:
#         if re.search(rf"\b{tag}\b", text, re.IGNORECASE):
#             found_tags.append(tag)
#     return found_tags if found_tags else ["general"]

# llm = getMassGpt()
# embedding_model = getLargeEmbedding()
# def getRetriever():
#     txt_documents = load_txts()
#     chunked_docs = []
#     for doc in txt_documents:
#         prompt = ChatPromptTemplate.from_messages([
#             ("system", 
#                 "You are a helpful assistant that splits medical or disease-related documents "
#                 "into semantically meaningful chunks for downstream semantic search and retrieval."
#             ),
#             ("human", 
#                 """You will be given a document. Split it into semantically coherent sections, following these rules:

#                 1. Do not omit any information.
#                 2. Keep related items together.
#                 3. Each chunk should be self-contained.
#                 4. Aim for ~800–1000 tokens per chunk (instead of 5000–6000 words).
#                 5. Return chunks as a numbered list with headings like '### Chunk 1:', '### Chunk 2:'.
#                 6. Include metadata (e.g., source, section) at the beginning of each chunk.

#                 Document:
#                 {document}
#                 """
#             )
#         ])

#         formatted_prompt = prompt.format_messages(document=doc.page_content)
#         response = llm.invoke(formatted_prompt)

#         chunks = [c.strip() for c in response.content.split("### Chunk") if c.strip()]
#         for idx, chunk in enumerate(chunks):
#             tags = extract_tags(chunk)
#             tags_str = ",".join(tags) if isinstance(tags, (list, set)) else str(tags)
#             enriched_metadata = {
#                 "source": doc.metadata.get("source", "unknown"),
#                 "doc_id": doc.metadata.get("doc_id", f"doc_{idx}"),
#                 "chunk_id": idx + 1,
#                 "title": doc.metadata.get("title", "untitled"),
#                 "tags":tags_str,
#                 "chunk_length": len(chunk.split()),
#                 "created_at": datetime.now().isoformat()
#             }
#             chunked_docs.append(Document(page_content=chunk, metadata=enriched_metadata))
#     vector_store = Chroma.from_documents(
#         chunked_docs,
#         embedding_model,
#         persist_directory="chroma_db"
#     )

#     retriever = vector_store.as_retriever(search_kwargs={"k": 3})
#     return retriever

# ----------------------------------------------------------
#         THIRD WAY
# ----------------------------------------------------------

from data.loader import load_txts
from utils.azure_llm import getMassGpt
from utils.embedding import getLargeEmbedding
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime
import re

llm = getMassGpt()
embedding_model = getLargeEmbedding()

def extract_tags(text: str):
    possible_tags = ["Causes", "Symptoms", "Diagnosis", "Treatment", "Prevention", "Complications", "Tests", "Triggers"]
    found_tags = []
    for tag in possible_tags:
        if re.search(rf"\b{tag}\b", text, re.IGNORECASE):
            found_tags.append(tag)
    return found_tags if found_tags else ["general"]

def getRetriever():
    # Load raw documents
    txt_documents = load_txts()

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
    # Build vector store
    vector_store = Chroma.from_documents(
        chunked_docs,
        embedding_model,
        persist_directory="chroma_db"
    )
    # Return retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    return retriever
