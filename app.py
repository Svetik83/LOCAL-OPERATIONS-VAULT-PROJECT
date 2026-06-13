import streamlit as st
import os
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Force crisp, modern dark-mode application styling
st.set_page_config(layout="wide", page_title="Local Operations Vault")
st.markdown("""<style>app_bg {background-color: #0E1117; color: #E0E0E0;}</style>""", unsafe_allow_html=True)

# Application core directory path mappings
UPLOAD_DIR = "./documents"
PERSIST_DIR = "./chroma_vault"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize local inference engines natively
@st.cache_resource
def load_local_engines():
    llm = Ollama(model="llama3", temperature=0.0)
    embeddings = OllamaEmbeddings(model="llama3")
    return llm, embeddings

llm, embeddings = load_local_engines()

# --- APPLICATION LAYOUT FRAME ---
st.title("🛡️ Offline Multi-Agent Compliance Swarm")
st.caption("AI Operations Portfolio Tier — Secured Local iMac Pro Hardware Environment")
st.write("---")

col1, col2 = st.columns([1, 2], gap="large")

# LEFT COLUMN: The Secure Document Vault Ingestion Tier
with col1:
    st.header("📁 Secure Document Vault")
    uploaded_file = st.file_uploader("Drop target administrative PDFs here", type=["pdf"])
    
    if uploaded_file:
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        with st.spinner("Processing ETL Chunking Splits..."):
            loader = PyPDFLoader(file_path)
            raw_docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
            chunks = text_splitter.split_documents(raw_docs)
            
            db = Chroma.from_documents(chunks, embeddings, persist_directory=PERSIST_DIR)
            st.success(f"✅ Ingested {len(chunks)} fragments successfully into disk!")

# RIGHT COLUMN: The Interactive Fact Canvas (Agent Swarm Console)
with col2:
    st.header("⚙️ Agent Swarm Interaction Canvas")
    user_query = st.text_input("Input your messy compliance query or administrative request:")
    
    if st.button("Trigger Swarm Process Pass") and user_query:
        if not os.path.exists(PERSIST_DIR) or not os.listdir(PERSIST_DIR):
            st.error("Error: The storage vault is currently empty. Ingest a document on the left first.")
        else:
            db = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
            retriever = db.as_retriever(search_kwargs={"k": 2})
            
            # --- EXECUTE LOOP ---
            with st.status("Swarm Orchestrator actively processing...", expanded=True) as status:
                
                # Agent 1 Run - Structured using Llama 3 Special Instruction Tokens
                st.write("🔄 Calling Agent 1: Optimizing database search parameters...")
                p_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a database keyword query generator. Extract only raw, space-separated search keywords from the text request.
STRICT CONFIGURATIONS: Output ONLY keyword tokens. Never include introductory conversational text, greeting sentences, bullet markers, formatting markdown, or punctuation.
Example Input: Check records to see if the asset tracker reports contain structural variations.
Example Output: asset tracker reports structural variations

User Request: {user_query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
                
                optimized_query = llm.invoke(p_prompt).strip()
                st.code(f"Target Parameters: {optimized_query}")
                
                # Agent 2 Run
                st.write("🔍 Calling Agent 2: Querying storage and evaluating raw evidence fragments...")
                docs = retriever.invoke(optimized_query)
                verified_facts = []
                for doc in docs:
                    g_prompt = f"Is this document relevant to '{optimized_query}'? Answer 'yes' or 'no'.\n\nCHUNK: {doc.page_content}"
                    if "yes" in llm.invoke(g_prompt).strip().lower():
                        verified_facts.append(doc.page_content)
                
                status.update(label="Analysis Loop Complete!", state="complete")
            
            # Agent 3 Compilation Presentation Run
            st.subheader("📊 Final Compiled Executive Briefing")
            if verified_facts:
                context_str = "\n".join([f"- {fact}" for fact in verified_facts])
                f_prompt = f"Compile a brief Markdown executive report based ONLY on these facts:\n{context_str}\n\nREQUEST: {user_query}"
                final_report = llm.invoke(f_prompt).strip()
                st.markdown(final_report)
            else:
                st.warning("⚠️ Guardrail Triggered: Information requested is not available inside the verified documents.")
