# import os
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.messages import HumanMessage
# from langchain_core.documents import Document
# from langchain_community.vectorstores import Chroma
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.runnables import RunnableMap
# from database.loader import load_pdfs, load_txts
# from azure_llm import getMassGpt
# from embedding import getLargeEmbedding


# def getRetriever():
#     full_text1 = load_pdfs()
#     full_text2 = load_txts()

#     combined_text = full_text1 + "\n\n" + full_text2

#     llm = getMassGpt()
#     embedding_model = getLargeEmbedding()

#     prompt = ChatPromptTemplate.from_messages([
#     ("system", "You are a helpful assistant that chunks documents for semantic search."),
#     ("human", """Split the following document into semantically meaningful sections. 
#         Follow these rules:
#         1. Do not omit any information — every detail must appear in some chunk.
#         2. Keep related items (like lists of dates, events, or tables) together in the same chunk.
#         3. Each chunk should be self-contained and understandable without needing other chunks.
#         4. Limit chunk size to about 5000–6000 words. If longer or Lesser, split carefully without breaking sentences or lists.
#         5. Return the chunks as a numbered list, with each chunk clearly separated.

#         Document:
#         {document}""")
#     ])

#     formatted_prompt = prompt.format_messages(document=combined_text)

#     response = llm.invoke(formatted_prompt)
#     chunks = response.content.split("\n\n")  # crude split; refine if needed

#     chunked_docs = [Document(page_content=chunk.strip()) for chunk in chunks if chunk.strip()]


#     vector_store = Chroma.from_documents(
#         chunked_docs,
#         embedding_model,
#         persist_directory="chroma_db"
#     )
#     # vector_store.persist()
#     retriever = vector_store.as_retriever(search_kwargs={"k": 3})
#     return retriever
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableMap
from database.loader import load_pdfs, load_txts
from utils.azure_llm import getMassGpt
from utils.embedding import getLargeEmbedding

def getRetriever():
    full_text1 = load_pdfs()
    full_text2 = load_txts()

    combined_texts = [
        {"text": full_text1, "source_id": "PDF_Doc1"},
        {"text": full_text2, "source_id": "TXT_Doc2"}
    ]

    llm = getMassGpt()
    embedding_model = getLargeEmbedding()

    chunked_docs = []
    for doc in combined_texts:
        # Use your existing chunking prompt
        prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that chunks documents for semantic search."),
        ("human", """Split the following document into semantically meaningful sections. 
            Follow these rules:
            1. Do not omit any information — every detail must appear in some chunk.
            2. Keep related items (like lists of dates, events, or tables) together in the same chunk.
            3. Each chunk should be self-contained and understandable without needing other chunks.
            4. Limit chunk size to about 5000–6000 words. If longer or Lesser, split carefully without breaking sentences or lists.
            5. Return the chunks as a numbered list, with each chunk clearly separated.

            Document:
            {document}""")
        ])
        formatted_prompt = prompt.format_messages(document=doc['text'])
        response = llm.invoke(formatted_prompt)
        chunks = response.content.split("\n\n")
        
        for idx, chunk in enumerate(chunks):
            if chunk.strip():
                chunked_docs.append(
                    Document(
                        page_content=chunk.strip(),
                        metadata={"source_id": f"{doc['source_id']}_chunk{idx+1}"}
                    )
                )

    vector_store = Chroma.from_documents(
        chunked_docs,
        embedding_model,
        persist_directory="chroma_db"
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    return retriever

# ----------------------------
### we tried this

#  prompt = ChatPromptTemplate.from_messages([
#             ("system", "You are a helpful assistant that chunks documents for semantic search."),
#             ("human", f"""Split the following document into semantically meaningful sections. 
#                 1. Do not omit any information.
#                 2. Keep related items together.
#                 3. Limit chunk size to about 5000–6000 words.
#                 Document:
#                 {doc['text']}""")
#         ])