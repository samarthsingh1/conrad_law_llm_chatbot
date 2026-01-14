# Conrad – AI Contract Review & Analysis Assistant

Conrad is an AI-first legal assistant designed to help review, search, and analyse long-form contracts.  
It uses Retrieval-Augmented Generation (RAG) built on top of LangChain, FAISS vector search, and a Chainlit UI to let users ask natural language questions about contracts and receive grounded, clause-level answers.

---

## 1. Project Goals

Conrad is built to:

- Reduce time spent manually reading long contracts.
- Surface relevant clauses and obligations given a free-form query.
- Provide concise, plain-language summaries of complex legal language.
- Enable experimentation with modern RAG patterns on real-world contract data (e.g. CUAD, ContractNLI, Contract-NLI style datasets).

This project is intended as a proof-of-concept for AI-assisted contract analysis, not a production legal tool.

---

## 2. Key Features

- **Retrieval-Augmented Generation (RAG)**  
  Uses FAISS as a vector store to retrieve the most relevant contract chunks before answering.

- **Semantic Clause Search**  
  Ask questions like “What are the termination rights?” or “Who bears data protection liability?” and Conrad will retrieve and quote the most relevant clauses.

- **Summarisation and Explanation**  
  Produces short, interpretable summaries of dense clauses to make contracts easier to understand.

- **Multi-Document Support**  
  Index and query multiple contracts or templates in one vector store.

- **Interactive Chat UI (Chainlit)**  
  Chat-style interface to interact with the assistant, test prompts, and refine questions in real time.

- **Optional Observability via LangSmith**  
  If configured, traces each run for debugging, prompt iteration, and evaluation.

---

## 3. Tech Stack

- **Language & Frameworks**
  - Python 3.10+
  - LangChain (RAG pipeline)
  - Chainlit (chat UI)
- **Vector Store**
  - FAISS (local, disk-persisted index)
- **LLM / Embeddings**
  - OpenAI models (configurable via environment variables)
- **Optional**
  - LangSmith for tracing and evaluation

---

## 4. Repository Structure

Your structure may differ slightly, but the core components are:

```text
conrad/
├─ app.py                         # Chainlit entrypoint
├─ ingest.py                      # Script to build the FAISS vector index from contract data
├─ requirements.txt               # Python dependencies
├─ .env.example                   # Example environment variables (no keys committed)
├─ notebook/
│  └─ vectorstores/
│     └─ contractnli_faiss/       # Persisted FAISS index files
├─ data/
│  ├─ raw/                        # Raw contract datasets (e.g. CUAD, ContractNLI)
│  └─ processed/                  # Cleaned / chunked contract text
└─ README.md
