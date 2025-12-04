from langchain_community.document_loaders import PyPDFLoader
import pandas as pd
import os

import pandas as pd

def load_inventory():
    file = r"C:\Users\GenaiblrpioUsr2\Desktop\Team24\database\data\inventory.csv"
    df = pd.read_csv(file)
    df.columns = [c.lower() for c in df.columns]
    inv = {}
    for _, r in df.iterrows():
        inv[r["item_id"].upper()] = {
            "name": r["product_name"],
            "stock": int(r["quantity"]),
            "threshold": int(r["threshold"]),
            "category": r["category"],
            "unit_price": r["unit_price"],
            "supplier": r["supplier"],
            "warehouse": r["warehouse"]
        }
    return inv


def load_sales_history():
    file = r"C:\Users\GenaiblrpioUsr2\Desktop\Team24\database\data\sales_history.csv"
    df = pd.read_csv(file)
    df.columns = [c.lower() for c in df.columns]
    return df

def load_pdfs():
    pdf_folder =  r"C:\Users\GenaiblrpioUsr2\Desktop\Team24\database\data\pdfs"
    pdf_files = [os.path.join(pdf_folder, f) for f in os.listdir(pdf_folder) if f.lower().endswith(".pdf")]

    # Load each PDF and store chunks
    all_docs = []

    for pdf_file in pdf_files:
        loader = PyPDFLoader(pdf_file)
        docs = loader.load()  # List[Document]
        
        # Optionally, add source info to each document
        for doc in docs:
            doc.metadata["source"] = os.path.basename(pdf_file)
            all_docs.append(doc)
            
    full_text = "\n".join([doc.page_content for doc in all_docs])
    return full_text


def load_txts():
    txt_folder = r"C:\Users\GenaiblrpioUsr2\Desktop\Team24\database\data\txts"
    txt_files = [os.path.join(txt_folder, f) for f in os.listdir(txt_folder) if f.lower().endswith(".txt")]

    all_docs = []

    for txt_file in txt_files:
        with open(txt_file, "r", encoding="utf-8") as f:
            content = f.read()

            # Create a simple document-like dict (similar to PDF loader output)
            doc = {
                "page_content": content,
                "metadata": {"source": os.path.basename(txt_file)}
            }
            all_docs.append(doc)

    # Combine all text into one big string
    full_text = "\n".join([doc["page_content"] for doc in all_docs])
    return full_text

def load_jsons():
    file_path = r"C:\Users\GenaiblrpioUsr2\Desktop\Team24\database\data\jsons\synthetic_data.json"
    # Open and load JSON
    with open(file_path, "r", encoding="utf-8") as f:
        trades = json.load(f)
    return trades


