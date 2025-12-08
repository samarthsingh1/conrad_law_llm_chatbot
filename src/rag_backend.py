from pathlib import Path
import os
import re

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

BASE_DIR = Path(__file__).resolve().parents[1] / "notebook"

USER_FAISS_DIR = BASE_DIR / "vectorstores" / "user_contracts_faiss"
CUAD_FAISS_DIR = BASE_DIR / "vectorstores" / "cuad_faiss_index"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

LLM_MODEL = "llama-3.3-70b-versatile"
# You can use a *smaller* model for routing below if you like.
ROUTER_MODEL = LLM_MODEL  # e.g. "llama-3.1-8b-instant" if available on Groq

GROQ_API_KEY = "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" 

# ============================================================
#  GLOBAL STATE
# ============================================================

embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

user_db = None     # FAISS index for user-uploaded contracts
cuad_db = None     # FAISS index for CUAD QA pairs

CURRENT_UPLOADED_CONTRACT = None


def format_top_k_clauses(retrieved_docs, k=5):
    """
    Utility for Chainlit debugging.
    Shows the top-k retrieved items with metadata,
    and distinguishes between CUAD QA and user clauses.
    """
    if not retrieved_docs:
        return " No retrieved items."

    retrieved_docs = retrieved_docs[:k]

    md = "## 🔎 Top Retrieved Items\n\n"

    for i, doc in enumerate(retrieved_docs, start=1):
        meta = doc.metadata or {}

        # Identify source
        if meta.get("contract_name"):
            source = "USER CONTRACT"
        else:
            source = "CUAD QA"

        chunk_id = meta.get("chunk_id", "N/A")

        # CUAD metadata
        q = meta.get("question", doc.page_content)
        a = meta.get("answer", None)
        ctx = meta.get("context", None)

        # USER contract metadata
        clause_no = meta.get("clause_number", None)

        md += f"### Result {i}\n"
        md += f"- **Source**: {source}\n"
        md += f"- **Chunk ID**: {chunk_id}\n"

        # USER CONTRACT formatting
        if source == "USER CONTRACT":
            preview = doc.page_content[:200].replace("\n", " ")
            md += f"- **Clause Number**: {clause_no}\n"
            md += f"- **Text Preview**: {preview}...\n\n"

        else:  # CUAD QA formatting
            md += f"- **Question**: {q}\n"
            if a:
                md += f"- **Answer**: {a}\n"
            if ctx:
                clean_ctx = ctx[:200].replace("\n", " ")
                md += f"- **Context (excerpt)**: {clean_ctx}...\n"
            md += "\n"

    return md

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
    """
    Load the CUAD QA-based FAISS DB.
    """
    global cuad_db
    if cuad_db is not None:
        return cuad_db

    if CUAD_FAISS_DIR.exists():
        print("📚 Loading CUAD QA FAISS DB...")
        cuad_db = FAISS.load_local(
            folder_path=str(CUAD_FAISS_DIR),
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        print("⚠️ CUAD FAISS index not found")
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
    """
    Parse a user-uploaded contract PDF into numbered clauses,
    and add them to the USER FAISS vectorstore.
    """
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
#  ROUTING LOGIC (HEURISTICS + LLM ROUTER)
# ============================================================

# Strong indicators that the user is asking about THEIR uploaded contract
CONTRACT_STRONG_KEYWORDS = [
    # explicit contract/agreement references
    "my contract",
    "this contract",
    "our contract",
    "my agreement",
    "this agreement",
    "our agreement",
    "uploaded contract",
    "uploaded agreement",
    "my nda",
    "this nda",
    "my lease",
    "this lease",

    # generic doc/file references that usually mean the uploaded PDF
    "this document",
    "in this document",
    "this file",
    "this pdf",
    "in the uploaded file",
    "in the uploaded document",
]

# Generic legal question phrasing (no explicit reference to user's contract)
GENERIC_QUESTION_PREFIXES = [
    "what is",
    "what are",
    "explain",
    "define",
    "how does",
    "how do",
    "typically",
    "in general",
    "usually",
]


DEICTIC_CONTRACT_HINTS = [
    " here",
    " here?",
    " here.",
    " in this clause",
    " in this section",
    " in this document",
    " in this agreement",
    " in this contract",
]

def _looks_like_contract_specific(q: str) -> bool:
    q = q.lower()

    # 1) Strong keywords: "my contract", "this agreement", etc.
    if any(kw in q for kw in CONTRACT_STRONG_KEYWORDS):
        return True

    # 2) Deictic hints like "here", "in this clause" WHEN a contract is uploaded
    if CURRENT_UPLOADED_CONTRACT and any(hint in q for hint in DEICTIC_CONTRACT_HINTS):
        return True

    # 3) Patterns like "clause 5", "section 3.2" together with "my/this/our"
    has_clause_ref = bool(
        re.search(r"\bclause\s+\d+(\.\d+)*", q) or
        re.search(r"\bsection\s+\d+(\.\d+)*", q)
    )
    has_possessive = any(w in q for w in ["my ", "this ", "our "])

    if has_clause_ref and has_possessive:
        return True

    return False



def _looks_like_generic_legal(q: str) -> bool:
    ql = q.lower()

    # Generic definitional question
    starts_like_def = any(
        ql.startswith(p) or f" {p} " in ql for p in GENERIC_QUESTION_PREFIXES
    )

    if starts_like_def and not any(kw in ql for kw in CONTRACT_STRONG_KEYWORDS):
        return True

    return False


def _create_router_llm():
    """Small LLM used only for routing when heuristics are ambiguous."""
    return ChatGroq(
        temperature=0,
        groq_api_key=GROQ_API_KEY,
        model=ROUTER_MODEL,
    )


def llm_route_decision(query: str) -> str:
    """
    Use an LLM to classify the query as:
      - 'contract' -> user uploaded contract
      - 'general'  -> generic legal / CUAD
      - 'both'     -> use both sources
    """
    router_llm = _create_router_llm()

    prompt = f"""
You are a routing classifier for a legal Q&A assistant.

Decide whether the USER is asking about:
- their SPECIFIC uploaded contract (label: contract),
- a GENERAL legal question (label: general), or
- BOTH (label: both).

Guidelines:
- Use 'contract' if they talk about "my contract", "this agreement", "this document",
  specific clause numbers in their document, etc.
- Use 'general' if they ask what something means in general, or typical legal meaning,
  without referencing their specific contract.
- Use 'both' if they clearly mix both (e.g. "in my contract and in general, how is X handled?").

Return ONLY one word: contract, general, or both.

User question: {query}

Answer with a single word:
"""

    resp = router_llm.invoke(prompt)
    label = resp.content.strip().lower()

    if "contract" in label:
        return "contract"
    if "both" in label:
        return "both"
    if "general" in label:
        return "general"

    # Fallback
    return "general"


def retrieve_docs(query: str, k: int = 6):
    """
    ROUTING LOGIC (IMPROVED):

    1. Heuristics first (cheap):
       - If clearly contract-specific → USER DB only.
       - If clearly generic legal → CUAD DB only.
    2. If ambiguous → LLM router:
       - 'contract' -> USER DB
       - 'general'  -> CUAD DB
       - 'both'     -> both DBs (split k)
    """
    q = query.strip()

    # 1) Heuristic rules
    if _looks_like_contract_specific(q):
        print("🟦 Routed to USER contract DB (lexical rule)")
        user_db = load_user_db()
        return user_db.similarity_search(q, k=k), []

    if _looks_like_generic_legal(q):
        print("🟩 Routed to CUAD legal QA DB (lexical rule)")
        kb_db = load_cuad_db()
        return [], kb_db.similarity_search(q, k=k)

    # 2) LLM router for ambiguous cases
    decision = llm_route_decision(q)
    print(f"🧠 LLM router decision: {decision}")

    if decision == "contract":
        print("🟦 Routed to USER contract DB (LLM router)")
        user_db = load_user_db()
        return user_db.similarity_search(q, k=k), []

    if decision == "general":
        print("🟩 Routed to CUAD legal QA DB (LLM router)")
        kb_db = load_cuad_db()
        return [], kb_db.similarity_search(q, k=k)

    # BOTH
    print("🟪 Routed to BOTH USER and CUAD DBs (LLM router)")
    user_db = load_user_db()
    kb_db = load_cuad_db()
    k_user = max(2, k // 2)
    k_kb = max(2, k - k_user)
    return (
        user_db.similarity_search(q, k=k_user),
        kb_db.similarity_search(q, k=k_kb),
    )


# ============================================================
#  LLM SETUP (MAIN ANSWER MODEL)
# ============================================================

def _create_llm():
    return ChatGroq(
        temperature=0,
        groq_api_key=GROQ_API_KEY,
        model=LLM_MODEL
    )


# ============================================================
#  CONTEXT FORMATTING FOR RAG
# ============================================================

def format_contract_docs(docs):
    if not docs:
        return "None"

    lines = []
    for d in docs:
        cname = d.metadata.get("contract_name", "User Contract")
        cnum = d.metadata.get("clause_number", "N/A")
        text = d.page_content.strip()
        lines.append(f"[Contract: {cname} | Clause {cnum}]\n{text}")
    return "\n\n---\n\n".join(lines)


def format_cuad_docs(docs):
    """
    Format CUAD QA documents for the prompt.

    Retrieval is done over "Question + Answer" embeddings,
    but we display them using structured metadata.
    """
    if not docs:
        return "None"

    MAX_CTX_CHARS = 1200

    lines = []
    for d in docs:
        meta = d.metadata or {}
        q = meta.get("question") or d.page_content
        a = meta.get("answer", "")
        ctx = meta.get("context", "")

        if len(ctx) > MAX_CTX_CHARS:
            ctx_display = ctx[:MAX_CTX_CHARS] + "... [truncated]"
        else:
            ctx_display = ctx

        lines.append(
            f"Question: {q}\n"
            f"Answer: {a}\n\n"
            f"Source contract excerpt:\n{ctx_display}"
        )
    return "\n\n---\n\n".join(lines)


# ============================================================
#  RAG CHAIN
# ============================================================

def get_rag_chain():
    llm = _create_llm()

    prompt = ChatPromptTemplate.from_template("""""")

    def prepare_context(q):
        user_docs, kb_docs = retrieve_docs(q)

        return {
            "contract_clauses": format_contract_docs(user_docs),
            "kb_clauses": format_cuad_docs(kb_docs),
            "question": q
        }

    return prepare_context | prompt | llm | StrOutputParser()



