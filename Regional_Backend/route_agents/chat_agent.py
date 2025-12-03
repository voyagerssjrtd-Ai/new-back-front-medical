from data.loader import json_to_text,get_patient_by_id
from utils.azure_llm import getMassGpt
from utils.chunking import getRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from utils.medPlusCalliing import getMediPlusRetrieverData

# retriever = getRetriever()
llm = getMassGpt()

def chat_query_agent(state):
    user_query = state["query"]
    retriever = getMediPlusRetrieverData(user_query)
    patient_record_json = get_patient_by_id("P001")
    patient_record = "Patient Past Record:\n" + "\n".join(json_to_text(patient_record_json)) if patient_record_json else "No past record found."

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
            "You are a helpful medical assistant. Answer questions using only the provided context. "
            "If the answer is not in the context, say you don't know rather than making up information."
        ),
        ("human",
            """You are given the patient's past medical records and the retrieved context from relevant documents.

            Context (retrieved from vector database):
            {context}

            Patient Past Record:
            {patient_record}

            Current patient query / symptom:
            {question}

            Guidelines:
            1. Use ONLY the information from the retrieved context and patient past record.
            2. Synthesize both sources to provide a clear, concise response.
            3. Do NOT invent facts not present in the context or past record.
            4. Provide your response strictly in **valid JSON format** with the following keys:

                - "Triage Level": "[Low / Medium / High / Emergency]"
                - "Reasoning": "[Explain why this triage level is assigned]"
                - "Urgent Evaluation Needed": "[Yes/No; specify tests if any]"
                - "Patient Actions": "[What patient should do next]"
                - "Clinician Tasks": "[Tasks clinician should perform]"
                - "Disclaimer": "[Include standard medical disclaimer]"

            5. If the context and past record do not contain the answer, respond:
                "The provided documents do not contain information about this."
            """
        )
    ])

    rag_chain = (
        {"context": retriever, "patient_record": lambda _: patient_record, "question": RunnablePassthrough()}
        | qa_prompt
        | llm
    )

    resp = rag_chain.invoke(user_query)
    state["output"] = resp.content
    return state