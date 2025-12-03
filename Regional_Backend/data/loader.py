
from langchain_community.document_loaders import PyPDFLoader
import pandas as pd
from langchain_core.documents import Document
import os
import json

def load_txts():
    txt_folder = r"C:\Users\GenAIBLRANCUSR33\Desktop\Backend\data\txt_files"
    txt_files = [os.path.join(txt_folder, f) for f in os.listdir(txt_folder) if f.lower().endswith(".txt")]

    all_docs = []
    for txt_file in txt_files:
        with open(txt_file, "r", encoding="utf-8") as f:
            content = f.read()
            # Return LangChain Document objects
            doc = Document(
                page_content=content,
                metadata={"source": os.path.basename(txt_file)}
            )
            all_docs.append(doc)
    return all_docs

def load_jsons():
    file_path = r"C:\Users\GenAIBLRANCUSR33\Desktop\Backend\data\json_files\mock_ehr.json"
    # Open and load JSON
    with open(file_path, "r", encoding="utf-8") as f:
        ehr_Report = json.load(f)
    return ehr_Report

def get_patient_by_id(patient_id):
    ehr_data = load_jsons()
    for patient in ehr_data.get("patients", []):
        if patient.get("id") == patient_id:
            return patient
    return None 

def json_to_text(d, prefix="- "):
    lines = []
    for key, value in d.items():
        key_name = key.replace("_", " ").title()
        if isinstance(value, dict):
            lines.append(f"{prefix}{key_name}:")
            lines.extend([f"  {line}" for line in json_to_text(value, prefix=prefix)])
        elif isinstance(value, list):
            if all(isinstance(i, dict) for i in value):
                lines.append(f"{prefix}{key_name}:")
                for item in value:
                    lines.extend([f"  {line}" for line in json_to_text(item, prefix=prefix)])
            else:
                lines.append(f"{prefix}{key_name}: {', '.join(map(str,value))}")
        else:
            lines.append(f"{prefix}{key_name}: {value}")
    return lines
