# 🛡️ Offline Multi-Agent Compliance Swarm

An enterprise-grade, zero-egress Retrieval-Augmented Generation (RAG) pipeline running a structured 3-agent optimization loop entirely on local workstation hardware. This architecture ingests sensitive administrative documents, executes automated relevance verification filtering to prevent LLM hallucinations, and compiles beautifully styled corporate briefings—completely offline with zero cloud API dependencies.

## 🏗️ Core Architecture & Workflow Lifecycle

The system separates concerns across three clear pipeline layers:

1. **Data Ingestion Tier (`src/ingestion.py`):** Automatically processes target PDF files from a local directory via `PyPDFDirectoryLoader`, applies a sliding character chunk layout (600-token window, 100-token overlap) to maximize Llama 3 context relevance, and populates a localized vector database directory (`./chroma_vault`).
2. **Deterministic Swarm Processing (`app.py` / `src/orchestrator.py`):**
   * **Agent 1 (The Planner):** Implements specialized prompt-wrapping and few-shot formatting rules to force the model to behave as a strict query token generator, stripping conversational filler out of raw user requests.
   * **Agent 2 (The Grader):** Evaluates retrieved document coordinates chunk-by-chunk using a binary relevance protocol, dynamically dropping out-of-bounds context.
   * **Agent 3 (The Formatter):** Processes the filtered fact matrix through the *Three Cs Protocol* (Critical verification, Creative parsing, Compassionate presentation) to construct the final output.
3. **Control Tower Interface (`app.py`):** A custom, responsive, dark-mode Streamlit split-screen application dividing local file staging actions on the left from real-time agent logging sequences on the right.

### 💻 Production Hardware Profile
* **Host Sandbox:** Workstation Environment (iMac Pro / 32 GB RAM)
* **Local Inference Engine:** Ollama running `llama3:8b` at a strict 0.0 temperature ceiling for absolute output determinism.
* **Storage Layer:** Localized ChromaDB vector persistence database.

---

## 🚀 Local Installation & Quickstart

### 1. Replicate the Sandbox Environment
Clone the repository and install the verified project packages:
```bash
git clone https://github.com/Svetik83/LOCAL-OPERATIONS-VAULT-PROJECT.git
cd LOCAL-OPERATIONS-VAULT-PROJECT
python3 -m venv ai_env
source ai_env/bin/activate
pip install -r requirements.txt
```

### 2. Set Up Local Models
Ensure Ollama is active on your host system and pull the target model requirements:
```bash
ollama pull llama3
```

### 3. Launch the App Canvas
Stage your administrative documents inside the folder directory and boot the local server layout:
```bash
streamlit run app.py
```
