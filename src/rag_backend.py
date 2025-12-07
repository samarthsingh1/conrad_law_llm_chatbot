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


# ============================================================
#  CUAD DATA LOADING & PROCESSING (QA-BASED)
# ============================================================

# def read_cuad_data():
#     """
#     Load CUAD JSON files into a Hugging Face dataset.
#     Expects files:
#       - train_separate_questions.json
#       - CUADv1.json
#       - test.json
#     under CUAD_DATA_DIR.
#     """
#     base_path = CUAD_DATA_DIR

#     dataset = load_dataset(
#         "json",
#         data_files={
#             "train": str(base_path / "train_separate_questions.json"),
#             "validation": str(base_path / "CUADv1.json"),
#             "test": str(base_path / "test.json"),
#         }
#     )
#     return dataset


# def process_cuad_data(dataset):
#     """
#     Flatten CUAD train split into:
#       - all_contracts: list of full contract texts (contexts)
#       - all_qas: list of dicts with question, answer, context, etc.

#     The CUAD 'train' split here is a single wrapper with a 'data' field.
#     """
#     train_item = dataset["train"][0]  # the single outer wrapper
#     all_contracts = []
#     all_qas = []

#     for entry in train_item["data"]:
#         for para in entry["paragraphs"]:
#             context = para["context"]
#             all_contracts.append(context)

#             for qa in para["qas"]:
#                 answer_text = qa["answers"][0]["text"] if qa["answers"] else ""
#                 answer_start = qa["answers"][0]["answer_start"] if qa["answers"] else -1

#                 all_qas.append(
#                     {
#                         "question": qa["question"],
#                         "answer": answer_text,
#                         "answer_start": answer_start,
#                         "context": context,
#                         "id": qa["id"],
#                     }
#                 )

#     return all_contracts, all_qas


# def build_cuad_qa_vectorstore(save_to_disk: bool = True):
#     """
#     Build a FAISS vectorstore from CUAD QA pairs.

#     IMPORTANT: Each document's *embedding* is based ONLY on the QUESTION
#     (page_content = question). The ANSWER and CONTEXT are stored in metadata
#     and are only used after retrieval, in the RAG prompt.
#     """
#     print("🧩 Building CUAD QA vectorstore from CUAD JSON...")
#     dataset = read_cuad_data()
#     _, all_qas = process_cuad_data(dataset)

#     docs = []
#     for qa in all_qas:
#         question_text = qa["question"]  # used for similarity search

#         docs.append(
#             Document(
#                 page_content=question_text,
#                 metadata={
#                     "qa_id": qa["id"],
#                     "answer": qa["answer"],
#                     "context": qa["context"],
#                     "answer_start": qa["answer_start"],
#                 }
#             )
#         )

#     db = FAISS.from_documents(docs, embeddings)

#     if save_to_disk:
#         CUAD_FAISS_DIR.mkdir(parents=True, exist_ok=True)
#         db.save_local(str(CUAD_FAISS_DIR))
#         print(f"✅ Saved CUAD QA FAISS index to {CUAD_FAISS_DIR}")

#     return db


# ============================================================
#  LOAD VECTOR DATABASES
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

    lines = []
    for d in docs:
        q = d.page_content
        a = d.metadata.get("answer", "")
        ctx = d.metadata.get("context", "")

        # Optional: truncate context if it's huge
        # ctx_display = ctx[:1500] + ("..." if len(ctx) > 1500 else "")
        ctx_display = ctx  # keep full for now

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
You are a legal reasoning assistant.

If the user asks about THEIR contract:
- Only use CONTRACT CLAUSES (uploaded user contract) as ground truth.
- Do NOT invent new terms or clauses that are not present in CONTRACT CLAUSES.

If the user asks a general legal question:
- Use KNOWLEDGE BASE Q&A PAIRS from the CUAD database.
- Treat them as examples of how similar questions are answered in real contracts.
- Clearly indicate that you are speaking about general legal practice, not the user's specific contract.

If both CONTRACT CLAUSES and KNOWLEDGE BASE CLAUSES are "None":
- Say you don't have enough information and ask the user to clarify or upload a contract.

---------------- CONTEXT ----------------
CONTRACT CLAUSES:
{contract_clauses}

KNOWLEDGE BASE CLAUSES (CUAD Q&A):
{kb_clauses}
-----------------------------------------

Question:
{question}

Instructions:
- First, state whether you are answering based on the user's contract or general CUAD legal knowledge.
- Give a clear, structured answer.
- When applicable, cite clause numbers and contract names for the user's contract, and refer to CUAD Q&A as examples.
""")

    def prepare_context(q):
        user_docs, kb_docs = retrieve_docs(q)

        return {
            "contract_clauses": format_contract_docs(user_docs),
            "kb_clauses": format_cuad_docs(kb_docs),
            "question": q
        }

    # LangChain Expression Language (LCEL) chain
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
