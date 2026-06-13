import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# Configuration matching your verified local system architecture
DATA_DIR = "./documents"
PERSIST_DIR = "./chroma_vault"
MODEL_NAME = "llama3"

def run_ingestion_pipeline():
    print("🚀 Initializing Local Data Ingestion Tier...")
    
    if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
        print(f"❌ Error: Place at least one PDF inside the '{DATA_DIR}' folder before running.")
        return

    # 1. Extract: Load all PDFs from the target directory
    print(f"📁 Scanning directory '{DATA_DIR}' for raw target assets...")
    loader = PyPDFDirectoryLoader(DATA_DIR)
    raw_documents = loader.load()
    print(f"📄 Extracted {len(raw_documents)} structural raw source pages.")

    # 2. Transform: Segment text into overlapping token windows
    print("✂️ Applying Recursive character chunking splits...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        length_function=len
    )
    doc_chunks = text_splitter.split_documents(raw_documents)
    print(f"🧩 Processed document text into {len(doc_chunks)} optimization fragments.")

    # 3. Load: Generate geometric coordinate coordinates and commit to database
    print(f"🧠 Vectorizing fragments via local model '{MODEL_NAME}' embeddings...")
    embeddings = OllamaEmbeddings(model=MODEL_NAME)
    
    print(f"💾 Saving mathematical vectors cleanly into disk vault: {PERSIST_DIR}")
    db = Chroma.from_documents(
        documents=doc_chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )
    print("✅ Ingestion Pipeline Complete! Secure Document Vault is populated.")

if __name__ == "__main__":
    run_ingestion_pipeline()
