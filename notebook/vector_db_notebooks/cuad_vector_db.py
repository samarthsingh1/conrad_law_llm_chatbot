from datasets import load_dataset
import os
import nltk
import re
from nltk.tokenize import sent_tokenize
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer


# ==============================================
# PATHS (UPDATED FOR CORRECT STORAGE)
# ==============================================

PROJECT_ROOT = "/Users/samarthsingh/PycharmProjects/conrad_law_llm_chatbot"
CUAD_DATA_DIR = f"{PROJECT_ROOT}/data/raw/cuad_data"

# Final FAISS save location
FAISS_SAVE_DIR = f"{PROJECT_ROOT}/notebook/vectorstores/cuad_faiss_index"


def read_data():
    dataset = load_dataset(
        "json",
        data_files={
            "train": f"{CUAD_DATA_DIR}/train_separate_questions.json",
            "validation": f"{CUAD_DATA_DIR}/CUADv1.json",
            "test": f"{CUAD_DATA_DIR}/test.json",
        }
    )
    return dataset


def process_data(dataset):
    train_item = dataset['train'][0]
    all_contracts = []
    all_qas = []

    for entry in train_item['data']:
        for para in entry['paragraphs']:
            context = para['context']
            all_contracts.append(context)

            for qa in para['qas']:
                answer_text = qa['answers'][0]['text'] if qa['answers'] else ""
                answer_start = qa['answers'][0]['answer_start'] if qa['answers'] else -1

                all_qas.append({
                    "question": qa['question'],
                    "answer": answer_text,
                    "answer_start": answer_start,
                    "context": context,
                    "id": qa['id']
                })

    return all_contracts, all_qas


def smart_clause_chunker(text):
    nltk.download('punkt')
    nltk.download('punkt_tab')

    if not text or not isinstance(text, str):
        return []

    text = re.sub(r"\n+", "\n", text)
    text = text.replace("\r", "").strip()

    split_by_heading = re.split(r"(?m)(?=^\s*\d+(\.\d+)*\s+)", text)

    clauses = []
    for block in split_by_heading:
        if not isinstance(block, str):
            continue

        block = block.strip()
        if len(block) < 20:
            continue

        try:
            sentences = sent_tokenize(block)
        except Exception:
            continue

        for i in range(0, len(sentences), 5):
            chunk = " ".join(sentences[i:i + 5]).strip()
            if len(chunk) > 30:
                clauses.append(chunk)

    return clauses


def chunk_meta_text(all_contracts):
    chunk_texts = []
    chunk_meta = []

    for c_idx, contract in enumerate(all_contracts):
        chunks = smart_clause_chunker(contract)
        for ch_idx, ch in enumerate(chunks):
            chunk_texts.append(ch)
            chunk_meta.append({
                "contract_id": c_idx,
                "chunk_id": ch_idx
            })

    return chunk_meta, chunk_texts


def emb_matrix_from_chunks(chunk_texts):
    model = SentenceTransformer("all-MiniLM-L6-v2")

    emb_matrix = model.encode(
        chunk_texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return emb_matrix


def build_vector_store(chunk_texts, chunk_meta, emb_matrix):
    emb = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True}
    )
    vector_store = FAISS.from_texts(
        texts=chunk_texts,
        embedding=emb,
        metadatas=chunk_meta
    )
    return vector_store


# ==============================================
# MAIN SCRIPT
# ==============================================

if __name__ == "__main__":
    dataset = read_data()
    all_contracts, all_qas = process_data(dataset)
    chunk_meta, chunk_texts = chunk_meta_text(all_contracts)
    emb_matrix = emb_matrix_from_chunks(chunk_texts)

    vector_store = build_vector_store(chunk_texts, chunk_meta, emb_matrix)

    print(f"\n Saving CUAD Vector DB to:\n{FAISS_SAVE_DIR}\n")
    vector_store.save_local(FAISS_SAVE_DIR)

    print(f"Processed {len(all_contracts)} contracts and {len(all_qas)} Q&A pairs.")
    print(" CUAD FAISS index successfully saved!")
