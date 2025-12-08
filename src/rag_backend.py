from pathlib import Path
import re

# -------- Prompts--------
from prompts import *
# -------- Vector DBs --------
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings

# -------- LLM --------
from langchain_groq import ChatGroq

# -------- Document Object --------
from langchain_core.documents import Document

# -------- Prompting & Parsing --------
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ============================================================
#  GLOBAL CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1] / "/Users/kanishkkaul/Desktop/NLP_Project/conrad_law_llm_chatbot/notebook"

USER_FAISS_DIR = BASE_DIR / "vectorstores" / "user_contracts_faiss"
CUAD_FAISS_DIR = BASE_DIR / "vectorstores" / "cuad_faiss_index"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

LLM_MODEL = "llama-3.3-70b-versatile"
GROQ_API_KEY = "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"  # replace with your real key

# ============================================================
#  GLOBAL STATE
# ============================================================

embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

user_db = None
cuad_db = None

CURRENT_UPLOADED_CONTRACT = None



# ============================================================
#  DEVELOPMENT TOOL (DELETE BEFORE PRODUCTION)
# ============================================================

def format_top_k_clauses(retrieved_docs, k=5):
    """
    Utility for Chainlit debugging.
    Shows the top-k retrieved chunks with metadata.
    """
    if not retrieved_docs:
        return " No retrieved clauses."

    retrieved_docs = retrieved_docs[:k]

    md = "### 🔎 Top Retrieved Clauses\n\n"

    for i, doc in enumerate(retrieved_docs, start=1):
        meta = doc.metadata or {}
        source = "USER CONTRACT" if meta.get("contract_name") else "CUAD KB"
        clause_no = meta.get("clause_number", "N/A")
        chunk_id = meta.get("chunk_id", "N/A")

        preview = doc.page_content[:200].replace("\n", " ") + "..."

        md += (
            f"**Result {i}**\n"
            f"- **Source:** `{source}`\n"
            f"- **Clause Number:** `{clause_no}`\n"
            f"- **Chunk ID:** `{chunk_id}`\n"
            f"- **Text Preview:** {preview}\n\n"
        )

    return md

# ============================================================
# LOAD VECTOR DATABASES
# ============================================================

def load_user_db():
    """Load the user-uploaded contract FAISS DB."""
    global user_db
    if user_db is None:
        print("📂 Loading USER contract FAISS DB...")
        user_db = FAISS.load_local(
            folder_path=str(USER_FAISS_DIR),
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
    return user_db


def load_cuad_db():
    """Load the CUAD legal knowledge base FAISS DB."""
    global cuad_db
    if cuad_db is None:
        print("📚 Loading CUAD FAISS DB...")
        cuad_db = FAISS.load_local(
            folder_path=str(CUAD_FAISS_DIR),
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
    return cuad_db


# ============================================================
#  CLAUSE EXTRACTION FROM USER PDF
# ============================================================

def extract_numbered_clauses(text: str):
    """
    Extract clauses based on patterns like:
    1
    1.1
    1.1.1
    2
    2.4
    """
    pattern = r"(?m)^(?P<num>\d+(\.\d+)*)(?P<body>[\s\S]*?)(?=^\d+(\.\d+)*|\Z)"
    clauses = []

    for m in re.finditer(pattern, text):
        clauses.append({
            "clause_number": m.group("num"),
            "text": (m.group("num") + " " + m.group("body")).strip()
        })
    return clauses


def add_contract_pdf_to_vectorstore(pdf_path: str, contract_name: str):
    """Parse PDF into clauses → store into FAISS user DB."""
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    full_text = "\n".join(p.page_content for p in pages)

    clauses = extract_numbered_clauses(full_text)
    print(f"🔍 Extracted {len(clauses)} clauses from contract PDF")

    docs = [
        Document(
            page_content=cl["text"],
            metadata={
                "contract_name": contract_name,
                "clause_number": cl["clause_number"],
                "chunk_id": idx
            }
        )
        for idx, cl in enumerate(clauses)
    ]

    db = load_user_db()

    print("FAISS BEFORE:", db.index.ntotal)
    db.add_documents(docs)
    db.save_local(str(USER_FAISS_DIR))
    print("FAISS AFTER:", db.index.ntotal)

    global CURRENT_UPLOADED_CONTRACT
    CURRENT_UPLOADED_CONTRACT = contract_name


async def process_pdf_and_add_to_vector_db(pdf_path: str):
    add_contract_pdf_to_vectorstore(pdf_path, Path(pdf_path).stem)


# ============================================================
#  ROUTING LOGIC (UNCHANGED - AS YOU REQUESTED)
# ============================================================

CONTRACT_KEYWORDS = [
    "my contract", "this contract", "uploaded contract",
    "clause", "section", "agreement", "provision", "term",
    "in the contract", "in my agreement"
]


def is_contract_question(query: str):
    q = query.lower()
    return any(kw in q for kw in CONTRACT_KEYWORDS)


def retrieve_docs(query: str, k=6):
    """
    ROUTING LOGIC — RETAINED EXACTLY AS YOU REQUESTED.

    If question references contract → search USER DB.
    Else → search CUAD DB.
    """
    if is_contract_question(query):
        print("🟦 Routed to USER contract DB")
        user_db = load_user_db()
        return user_db.similarity_search(query, k=k), []

    print("🟩 Routed to CUAD legal KB DB")
    kb_db = load_cuad_db()
    return [], kb_db.similarity_search(query, k=k)


# ============================================================
#  LLM SETUP
# ============================================================

def _create_llm():
    return ChatGroq(
        temperature=0,
        groq_api_key=GROQ_API_KEY,
        model=LLM_MODEL
    )




def get_rag_chain():
    llm = _create_llm()
    # 4) COMBINED PROMPT
    prompt = ChatPromptTemplate.from_messages([
        ("system", global_system_prompt),
        ("system", user_vector_db_prompt),
        ("system", cuad_vector_db_prompt),
        (
            "human",
            """You are now answering a single user question.

--------------------------
CONTRACT_CLAUSES (uploaded contract text, may be "None"):
{contract_clauses}

--------------------------
KB_CLAUSES (general legal background or CUAD snippets, may be "None"):
{kb_clauses}

--------------------------
USER QUESTION:
{question}

INSTRUCTIONS:
- Infer whether you are in USER CONTRACT MODE (contract_clauses present) or CUAD MODE (kb_clauses present, contract_clauses empty).
- Classify the question internally as one of: Fetching, Verification, Reasoning, Simple factual Q&A.
- Follow the mode-specific behavior and output format from the system messages.
- Do NOT mention the internal labels or your reasoning steps.
- Just provide the final answer in the appropriate structured format.
"""
        ),
    ])

    def prepare_context(q: str):
        # Existing routing logic: sends to user_db if CONTRACT_KEYWORDS match, else CUAD.
        user_docs, kb_docs = retrieve_docs(q)

        contract_text = "\n\n".join(d.page_content for d in user_docs) or "None"
        kb_text = "\n\n".join(d.page_content for d in kb_docs) or "None"

        return {
            "contract_clauses": contract_text,
            "kb_clauses": kb_text,
            "question": q,
        }

    return prepare_context | prompt | llm | StrOutputParser()
