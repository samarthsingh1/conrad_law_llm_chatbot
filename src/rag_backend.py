from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


def get_rag_chain():

    # ---------------------------------------
    # Paths - matches your screenshot exactly
    # ---------------------------------------
    base_dir = Path(__file__).resolve().parents[1] / "notebook"
    faiss_store_dir = base_dir / "vectorstores" / "contractnli_faiss"

    print("Loading FAISS from:", faiss_store_dir)

    # ---------------------------------------
    # Embeddings
    # ---------------------------------------
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # ---------------------------------------
    # Load FAISS index
    # ---------------------------------------
    db = FAISS.load_local(
        folder_path=str(faiss_store_dir),
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = db.as_retriever(search_kwargs={"k": 4})
    print("FAISS retriever ready.")

    # ---------------------------------------
    # HF LLM (CPU-friendly)
    # ---------------------------------------
    model_name = "google/flan-t5-base"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    hf_pipeline = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512,
        temperature=0.0
    )

    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    # ---------------------------------------
    # Prompt
    # ---------------------------------------
    prompt = ChatPromptTemplate.from_template("""
    You are a legal contract analysis assistant.

    Use ONLY the context provided below to answer the user question.
    If the answer is not found in the retrieved clauses, say:
    "The contract does not contain this information."

    Context:
    {context}

    Question:
    {question}

    Answer in clear legal language.
    """)

    # ---------------------------------------
    # Format docs
    # ---------------------------------------
    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    # ---------------------------------------
    # Build RAG chain
    # ---------------------------------------
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    print("RAG chain ready.")

    return rag_chain
