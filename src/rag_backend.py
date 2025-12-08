from pathlib import Path
import os
import re

# -------- External Data --------
# from datasets import load_dataset

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

# Folder where CUAD JSON files live:
#   - train_separate_questions.json
#   - CUADv1.json
#   - test.json
CUAD_DATA_DIR = BASE_DIR / "cuad_data"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

LLM_MODEL = "llama-3.3-70b-versatile"
GROQ_API_KEY = "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"  # replace with your real key

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

        # CUAD QA fields
        q = doc.page_content
        a = meta.get("answer", None)
        ctx = meta.get("context", None)

        # User contract fields
        clause_no = meta.get("clause_number", None)

        md += f"### Result {i}\n"
        md += f"- **Source**: {source}\n"

        if source == "USER CONTRACT":
            md += f"- **Clause Number**: {clause_no}\n"
            preview = doc.page_content[:200].replace("\n", " ")
            md += f"- **Text Preview**: {preview}...\n\n"

        else:  # CUAD QA
            md += f"- **Question**: {q}\n"
            if a:
                md += f"- **Answer**: {a}\n"
            if ctx:
                md += f"- **Context (excerpt)**: {ctx[:200].replace('\n', ' ')}...\n"
            md += "\n"

    return md


def load_cuad_db():
    """
    Load the CUAD QA-based FAISS DB.

    If it does not exist yet, build it from the CUAD QA pairs.
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
        # cuad_db = build_cuad_qa_vectorstore(save_to_disk=True)
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

    Each clause captures from its number until the next number or end of text.
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
    """
    Async wrapper for adding a user contract PDF to the FAISS DB.
    Can be awaited from an async web framework (e.g., FastAPI, Streamlit async).
    """
    add_contract_pdf_to_vectorstore(pdf_path, Path(pdf_path).stem)


# ============================================================
#  ROUTING LOGIC (UNCHANGED IN SPIRIT)
# ============================================================

CONTRACT_KEYWORDS = [
    "my contract", "this contract", "uploaded contract",
    "clause", "section", "agreement", "provision", "term",
    "in the contract", "in my agreement"
]


def is_contract_question(query: str):
    """
    Heuristic routing: if query references 'my contract' or similar,
    we treat it as a question about the user-uploaded contract.
    """
    q = query.lower()
    return any(kw in q for kw in CONTRACT_KEYWORDS)


def retrieve_docs(query: str, k=6):
    """
    ROUTING LOGIC:

    - If question references the user's contract → search USER DB only.
    - Otherwise → search CUAD QA DB only.

    Returns:
      (user_docs, kb_docs)
    """
    if is_contract_question(query):
        print("🟦 Routed to USER contract DB")
        user_db = load_user_db()
        return user_db.similarity_search(query, k=k), []

    print("🟩 Routed to CUAD legal QA DB")
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


# ============================================================
#  CONTEXT FORMATTING FOR RAG
# ============================================================

def format_contract_docs(docs):
    """
    Format user contract clauses for the prompt.
    Uses metadata to show contract name & clause number where available.
    """
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

    Remember: for CUAD, page_content = QUESTION.
    ANSWER + CONTEXT are stored in metadata and reconstructed here.
    """
    if not docs:
        return "None"

    MAX_CTX_CHARS = 1200  # <-- keep this small; adjust if needed

    lines = []
    for d in docs:
        q = d.page_content
        a = d.metadata.get("answer", "")
        ctx = d.metadata.get("context", "")

        # 🔹 Truncate long contexts aggressively
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

    prompt = ChatPromptTemplate.from_template("""
You are a careful legal reasoning assistant. You MUST base your answer ONLY on the context provided below.

---------------- CONTRACT CONTEXT ----------------
CONTRACT CLAUSES (User-uploaded contract):
{contract_clauses}

KNOWLEDGE BASE Q&A (CUAD dataset):
{kb_clauses}
--------------------------------------------------

User question:
{question}

ROLE & CONTEXT USAGE
- If the answer can be found in the user's CONTRACT CLAUSES (e.g., question mentions "my contract", "this agreement", "clause X", "section Y"):
  - Treat CONTRACT CLAUSES as ground truth.
  - Use CUAD Q&A only as background knowledge if absolutely needed, and clearly label it as "general practice", not binding on the user's contract.
- If the question is a general legal question (not about a specific uploaded contract):
  - Answer using CUAD Q&A as examples of how similar questions are answered in real contracts.
  - DO NOT invent laws; stay within the patterns in CUAD.

ANSWER FORMAT (VERY IMPORTANT)
Always answer in **markdown** with the following structure:

1. ## Answer summary
   - 2–4 bullet points that give a concise, non-technical summary.
2. ## Detailed explanation
   - 2–5 short paragraphs explaining the concept in plain English.
   - When explaining a term (e.g. "force majeure"), include:
     - What it generally means.
     - Typical conditions for it to apply.
     - At least one simple real-world example.
3. ## Relevant clauses / examples
   - If using the user's contract:
     - Bullet points like:
       - **Clause 5.2 – Force Majeure:** [brief paraphrase]
       - **Clause 9.1 – Termination for Force Majeure:** [brief paraphrase]
   - If using CUAD Q&A:
     - Bullet points like:
       - **Example 1 (CUAD QA):** [short paraphrase of answer]
       - **Contract excerpt:** [one-sentence summary of the context]
4. ## Caveats
   - 1–3 bullet points noting:
     - That this is not formal legal advice.
     - That exact rights and obligations depend on the full contract and jurisdiction.

STYLE GUIDELINES
- Use clear headings and bullet points.
- Keep sentences medium length; avoid long, dense paragraphs.
- Quote or paraphrase relevant clauses instead of copying huge blocks of text.
- If information is missing in the context, say so explicitly instead of guessing.

Now write the answer following the structure above.
""")

    def prepare_context(q):
        user_docs, kb_docs = retrieve_docs(q)

        return {
            "contract_clauses": format_contract_docs(user_docs),
            "kb_clauses": format_cuad_docs(kb_docs),
            "question": q
        }

    return prepare_context | prompt | llm | StrOutputParser()


def answer_question(question: str) -> str:
    """
    Convenience wrapper: call this function from your UI / notebook.

    Example:
        response = answer_question("What is the non-compete clause in my contract?")
    """
    chain = get_rag_chain()
    return chain.invoke(question)


# ============================================================
#  OPTIONAL: SIMPLE MANUAL TEST
# ============================================================

if __name__ == "__main__":
    # Example manual tests (you can comment these out in production)

    # General legal question → CUAD QA DB
    q1 = "What is a typical non-compete clause in a SaaS agreement?"
    print("Q1:", q1)
    print("A1:", answer_question(q1))
    print("\n" + "=" * 80 + "\n")

    # Contract-specific question → USER DB
    # (Requires that you have already built USER_FAISS_DIR separately)
    q2 = "What does the termination clause in my contract say?"
    print("Q2:", q2)
    print("A2:", answer_question(q2))
