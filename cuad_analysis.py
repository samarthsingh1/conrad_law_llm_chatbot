from datasets import load_dataset
import os
import nltk
import re
from nltk.tokenize import sent_tokenize
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer


def read_data():
    # Always build absolute paths safely
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.join(script_dir, "cuad_data")

    dataset = load_dataset(
        "json",
        data_files={
            "train": os.path.join(base_path, "train_separate_questions.json"),
            "validation": os.path.join(base_path, "CUADv1.json"),
            "test": os.path.join(base_path, "test.json"),
        }
    )
    return dataset

def process_data(dataset):
    train_item = dataset['train'][0]  # the single outer wrapper
    all_contracts = []  # list of full contract texts
    all_qas = []        # list of dicts: {question, answer, answer_start, context}

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
        return []  # return empty chunk list if text is None or not a string

    # 1. Normalize spacing
    text = re.sub(r"\n+", "\n", text)
    text = text.replace("\r", "").strip()

    # 2. Split by legal headings (some results may be None)
    split_by_heading = re.split(r"(?m)(?=^\s*\d+(\.\d+)*\s+)", text)

    clauses = []

    for block in split_by_heading:
        if not block or not isinstance(block, str):
            continue  # skip None or empty blocks

        block = block.strip()
        if len(block) < 20:
            continue  # skip tiny junk blocks

        # 3. Break by sentences for large blocks
        sentences = []
        try:
            sentences = sent_tokenize(block)
        except Exception:
            continue  # skip if tokenizer fails

        # 4. Merge sentences into clause chunks (5-sentence windows)
        for i in range(0, len(sentences), 5):
            chunk = " ".join(sentences[i:i+5]).strip()

            if len(chunk) > 30:  # keep meaningful chunks only
                clauses.append(chunk)

    return clauses

def test_clause_chunker(all_contracts):   
    all_chunks = []

    for contract in all_contracts:
        all_chunks.extend(smart_clause_chunker(contract))
    return all_chunks   

def chunk_meta_text(all_contracts):
    chunk_texts = []
    chunk_meta = []   # to store contract + chunk indices

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
        normalize_embeddings=True  # cosine similarity friendly
    )

    return emb_matrix

def faiss_index_from_emb_matrix(emb_matrix):
    d = emb_matrix.shape[1]
    index = faiss.IndexFlatIP(d)      # inner-product index
    index.add(emb_matrix)             # add vectors
    return index

def search(chunk_texts,chunk_meta, query, index, k):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    q_emb = model.encode([query], normalize_embeddings=True)
    scores, idxs = index.search(q_emb, k)  # (1, k)
    idxs = idxs[0]
    scores = scores[0]
    
    results = []
    for rank, (i, s) in enumerate(zip(idxs, scores), start=1):
        results.append({
            "rank": rank,
            "score": float(s),
            "text": chunk_texts[i],
            "meta": chunk_meta[i],
        })
    return results

def build_vector_store(chunk_texts, chunk_meta, emb_matrix):
    emb = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True}
    )
    vector_store = FAISS.from_texts(texts=chunk_texts, embedding=emb, metadatas=chunk_meta)
    return vector_store


if __name__ == "__main__":
    dataset = read_data()
    all_contracts, all_qas = process_data(dataset)
    all_chunks = test_clause_chunker(all_contracts)
    chunk_meta, chunk_texts = chunk_meta_text(all_contracts)
    emb_matrix = emb_matrix_from_chunks(chunk_texts)
    index = faiss_index_from_emb_matrix(emb_matrix)

    vector_store = build_vector_store(chunk_texts, chunk_meta, emb_matrix)
    vector_store.save_local("cuad_faiss_index")

    print(f"Processed {len(all_contracts)} contracts and {len(all_qas)} Q&A pairs.")
    # quick test
    query = "Explain indemnification obligations"
    for r in search(chunk_texts, chunk_meta, query, index, k=3):
        print("\n[RANK", r["rank"], "SCORE", round(r["score"], 3), "]")
        print(r["text"][:500], "...")
    