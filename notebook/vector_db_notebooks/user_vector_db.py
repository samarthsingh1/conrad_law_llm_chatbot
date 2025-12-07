# init_user_faiss.py
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Adjust this if your path is different
BASE_DIR = Path("/notebook")
FAISS_STORE_DIR = BASE_DIR / "vectorstores" / "cuad_contracts_faiss"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

print(f"Creating NEW FAISS store at: {FAISS_STORE_DIR}")

FAISS_STORE_DIR.mkdir(parents=True, exist_ok=True)

embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    encode_kwargs={"normalize_embeddings": True},
)

# We need at least one vector to initialize FAISS.
texts = ["__INITIAL_PLACEHOLDER__"]
metadatas = [{"contract_name": "__placeholder__", "chunk_id": 0}]

db = FAISS.from_texts(
    texts=texts,
    embedding=embeddings,
    metadatas=metadatas,
)

db.save_local(str(FAISS_STORE_DIR))
print(" Empty user FAISS index initialized.")
