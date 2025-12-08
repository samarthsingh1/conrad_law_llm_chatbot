# notebook/vector_db_notebooks/cuad_vector_db.py

from pathlib import Path
from datasets import load_dataset
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


# ==============================================
# PATHS (DYNAMIC, MATCH REPO LAYOUT)
# ==============================================

# This file is in: <repo_root>/notebook/vector_db_notebooks/cuad_vector_db.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Raw CUAD JSONs:
#   <repo_root>/data/raw/cuad_data/{train_separate_questions.json, CUADv1.json, test.json}
CUAD_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "cuad_data"

# Final FAISS save location:
#   <repo_root>/notebook/vectorstores/cuad_faiss_index
FAISS_SAVE_DIR = PROJECT_ROOT / "notebook" / "vectorstores" / "cuad_faiss_index"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ==============================================
# CUAD DATA LOADING & PROCESSING
# ==============================================

def read_data():
    dataset = load_dataset(
        "json",
        data_files={
            "train": str(CUAD_DATA_DIR / "train_separate_questions.json"),
            "validation": str(CUAD_DATA_DIR / "CUADv1.json"),
            "test": str(CUAD_DATA_DIR / "test.json"),
        },
    )
    return dataset


def process_data(dataset):
    """
    Flatten CUAD train split into:
      - all_contracts: list of contexts (full contract text)
      - all_qas: list of dicts with question, answer, context, etc.

    IMPORTANT:
    The 'train' split contains multiple items, one per question type
    (Exclusivity, Indemnification, etc.). We must iterate over ALL of them,
    not just dataset['train'][0].
    """
    all_contracts = []
    all_qas = []

    # 🔹 Iterate over ALL train items (each is a question family)
    for train_item in dataset["train"]:
        for entry in train_item["data"]:
            for para in entry["paragraphs"]:
                context = para["context"]
                all_contracts.append(context)

                for qa in para["qas"]:
                    answer_text = qa["answers"][0]["text"] if qa["answers"] else ""
                    answer_start = qa["answers"][0]["answer_start"] if qa["answers"] else -1

                    all_qas.append(
                        {
                            "question": qa["question"],
                            "answer": answer_text,
                            "answer_start": answer_start,
                            "context": context,
                            "id": qa["id"],
                        }
                    )

    return all_contracts, all_qas


# ==============================================
# BUILD QA-BASED VECTOR STORE
# ==============================================

def build_qa_vector_store(all_qas):
    """
    Build a FAISS vector store where:
      - page_content = "Question: ...\\nAnswer: ..."
        (this is what gets embedded for similarity search)
      - metadata = {question, answer, context, qa_id, answer_start}
        (used later for prompt formatting)
    """
    print(f"Building CUAD QA vector store from {len(all_qas)} QA pairs...")

    texts = []
    metadatas = []

    for qa in all_qas:
        question_text = qa["question"] or ""
        answer_text = qa["answer"] or ""

        # 🔹 This combined string is what FAISS will embed & compare to user queries
        combined = f"Question: {question_text}\nAnswer: {answer_text}"

        texts.append(combined)
        metadatas.append(
            {
                "qa_id": qa["id"],
                "question": question_text,
                "answer": answer_text,
                "context": qa["context"],
                "answer_start": qa["answer_start"],
            }
        )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )

    vector_store = FAISS.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
    )
    return vector_store


# ==============================================
# MAIN SCRIPT
# ==============================================

if __name__ == "__main__":
    print(f"Project root: {PROJECT_ROOT}")
    print(f"CUAD data dir: {CUAD_DATA_DIR}")
    print(f"FAISS will be saved to: {FAISS_SAVE_DIR}")

    dataset = read_data()
    all_contracts, all_qas = process_data(dataset)

    vector_store = build_qa_vector_store(all_qas)

    FAISS_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(FAISS_SAVE_DIR))

    print("\nCUAD QA-based FAISS index successfully saved!")
    print(f"Contracts processed: {len(all_contracts)}")
    print(f"QA pairs processed: {len(all_qas)}")
