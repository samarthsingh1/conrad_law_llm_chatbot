from pathlib import Path
import re

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from langchain_core.documents import Document

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# -----------------------------------------------------
# GLOBAL CONFIG
# -----------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1] / "notebook"
FAISS_STORE_DIR = BASE_DIR / "vectorstores" / "contractnli_faiss"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Groq LLM model + API KEY (hardcoded as you requested)
LLM_MODEL = "llama-3.3-70b-versatile"
GROQ_API_KEY = "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"

# Tracks current uploaded contract
CURRENT_UPLOADED_CONTRACT = None

# -----------------------------------------------------
# GLOBAL STATE
# -----------------------------------------------------

embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
db = None


# -----------------------------------------------------
# LOAD & SAVE FAISS INDEX
# -----------------------------------------------------

def _load_faiss_index():
    """Load FAISS index once, cached."""
    global db

    if db is None:
        print("\n🔍 Loading FAISS index from:", FAISS_STORE_DIR)

        db = FAISS.load_local(
            folder_path=str(FAISS_STORE_DIR),
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )

        print(" FAISS index loaded.")

    return db


# -----------------------------------------------------
# PARAGRAPH-BASED CHUNKING
# -----------------------------------------------------

def split_into_paragraphs(docs):
    """Split PDF into natural paragraph chunks."""
    paragraphs = []

    for doc in docs:
        text = doc.page_content.replace("\r", "\n")
        raw_parts = re.split(r"\n\s*\n", text)

        for p in raw_parts:
            cleaned = p.strip()
            if len(cleaned) < 60:
                continue
            paragraphs.append(cleaned)

    print(f"📄 Paragraph chunks created: {len(paragraphs)}")
    return paragraphs


# -----------------------------------------------------
# INGEST CONTRACT PDF (USER GROUND TRUTH)
# -----------------------------------------------------


def extract_numbered_clauses(text: str):
    """
    Split contract text into numbered clauses.
    Supports:
      1
      1.1
      1.1.1
      2
      2.4
    """
    # Regex to capture clause headings
    pattern = r"(?m)^(?P<num>\d+(\.\d+)*)(?P<body>[\s\S]*?)(?=^\d+(\.\d+)*|\Z)"

    matches = re.finditer(pattern, text)

    clauses = []
    for m in matches:
        full_clause = (m.group("num") + " " + m.group("body")).strip()
        clauses.append({
            "clause_number": m.group("num"),
            "text": full_clause
        })

    return clauses


def add_contract_pdf_to_vectorstore(pdf_path: str, contract_name: str):
    print(f"\n📥 Ingesting PDF USING CLAUSE SPLITTING: {pdf_path}")

    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    # Combine full PDF text
    full_text = "\n".join(p.page_content for p in pages)

    # Extract legal clauses
    clauses = extract_numbered_clauses(full_text)
    print(f"📚 Extracted clauses: {len(clauses)}")

    # Convert to LangChain Document objects
    docs = []
    for idx, cl in enumerate(clauses):
        clause_text = cl["text"]
        clause_number = cl["clause_number"]

        docs.append(
            Document(
                page_content=clause_text,
                metadata={
                    "contract_name": contract_name,
                    "clause_number": clause_number,
                    "chunk_id": idx
                }
            )
        )

    faiss_db = _load_faiss_index()

    print(" FAISS BEFORE:", faiss_db.index.ntotal)

    faiss_db.add_documents(docs)
    faiss_db.save_local(str(FAISS_STORE_DIR))

    print(" FAISS AFTER:", faiss_db.index.ntotal)

    global CURRENT_UPLOADED_CONTRACT
    CURRENT_UPLOADED_CONTRACT = contract_name

    print("✅ CLAUSE-LEVEL PDF ingestion complete.")


# Chainlit helper
async def process_pdf_and_add_to_vector_db(pdf_path: str):
    add_contract_pdf_to_vectorstore(pdf_path, Path(pdf_path).stem)
    print("✅ PDF ingestion complete.")


# -----------------------------------------------------
# RETRIEVAL (PDF priority → fallback KB)
# -----------------------------------------------------

def retrieve_with_pdf_priority(query: str, faiss_db, k_pdf=5, k_global=10):
    """Retrieve PDF clauses first, KB clauses second."""
    pdf_results = []
    if CURRENT_UPLOADED_CONTRACT:
        pdf_results = faiss_db.similarity_search(
            query, k=k_pdf,
            filter={"contract_name": CURRENT_UPLOADED_CONTRACT}
        )

    kb_results = faiss_db.similarity_search(query, k=k_global)

    # de-dupe, PDF always has priority
    seen = set()
    final = []

    for doc in (pdf_results + kb_results):
        key = doc.page_content[:80]
        if key not in seen:
            final.append(doc)
            seen.add(key)

    return final


# -----------------------------------------------------
# LLM CREATION
# -----------------------------------------------------

def _create_llm():
    print("\n🤖 Loading ChatGroq:", LLM_MODEL)
    return ChatGroq(
        temperature=0,
        groq_api_key=GROQ_API_KEY,
        model=LLM_MODEL
    )


# -----------------------------------------------------
# MAIN RAG CHAIN (Clause Display + Answer)
# -----------------------------------------------------

def get_rag_chain():
    faiss_db = _load_faiss_index()
    llm = _create_llm()

    print("🔎 Retrieval: Contract PDF first -> KB fallback")

    prompt = ChatPromptTemplate.from_template("""
You are a legal contract analysis assistant.

GROUND TRUTH must always come from the uploaded contract’s retrieved clauses.
Only when the clause does NOT appear in the uploaded contract may you reference the external knowledge base.

CONTRACT CLAUSES (PRIMARY SOURCE):
{contract_clauses}

KNOWLEDGE BASE CLAUSES (SECONDARY SOURCE):
{kb_clauses}

Question:
{question}

Respond with concise legal reasoning, cite clause numbers when visible.
""")

    def prepare_context(q):
        docs = retrieve_with_pdf_priority(q, faiss_db)

        pdf_clauses, kb_clauses = [], []

        for d in docs:
            if d.metadata.get("contract_name") == CURRENT_UPLOADED_CONTRACT:
                pdf_clauses.append(d)
            else:
                kb_clauses.append(d)

        contract_md = ""
        for i, d in enumerate(pdf_clauses):
            meta = d.metadata
            contract_md += f"📌 **Contract Clause {i+1}** (chunk {meta['chunk_id']}):\n{d.page_content}\n\n"

        kb_md = ""
        for i, d in enumerate(kb_clauses):
            kb_md += f"📚 KB Clause {i+1}:\n{d.page_content}\n\n"

        return {
            "contract_clauses": contract_md or "No contract clauses retrieved.",
            "kb_clauses": kb_md or "No KB clauses found.",
            "question": q
        }

    rag_chain = (
        prepare_context
        | prompt
        | llm
        | StrOutputParser()
    )

    print("🧩 RAG chain initialized.")
    return rag_chain
