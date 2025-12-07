import chainlit as cl
from pathlib import Path

from rag_backend import (
    get_rag_chain,
    process_pdf_and_add_to_vector_db,
    retrieve_with_pdf_priority,
    _load_faiss_index,
    CURRENT_UPLOADED_CONTRACT
)

# Global RAG chain
rag_chain = get_rag_chain()


@cl.on_chat_start
async def on_chat_start():
    await cl.Message(
        content=(
            "**📑 Contract RAG Assistant**\n\n"
            "1. Upload a contract PDF (NDA, service agreement, etc).\n"
            "2. After it is processed, ask questions like:\n"
            "   - \"What is the termination clause?\"\n"
            "   - \"How is confidentiality defined?\"\n"
            "   - \"What happens if either party breaches the agreement?\"\n\n"
            "The uploaded contract is treated as **ground truth**.\n"
            "The reference knowledge base is only used as backup."
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """
    Handles:
    - PDF uploads
    - Running the RAG chain
    - Displaying retrieved clauses
    """
    global rag_chain

    # -----------------------------------------------------
    # 1) HANDLE PDF UPLOADS
    # -----------------------------------------------------
    if message.elements:
        for element in message.elements:
            if isinstance(element, cl.File) and element.path.lower().endswith(".pdf"):
                pdf_path = element.path
                file_name = element.name or Path(pdf_path).name

                await cl.Message(
                    content=f"📄 Processing uploaded PDF: **{file_name}**"
                ).send()

                await process_pdf_and_add_to_vector_db(pdf_path)

                # Rebuild RAG chain with updated vector DB
                rag_chain = get_rag_chain()

                await cl.Message(
                    content=f"✅ **{file_name}** has been ingested.\n\nYou can now ask questions about this contract."
                ).send()

        if not message.content or message.content.strip() == "":
            return

    # -----------------------------------------------------
    # 2) PROCESS QUESTION
    # -----------------------------------------------------
    question = (message.content or "").strip()
    if not question:
        await cl.Message(content="Please enter a question about the contract.").send()
        return

    await cl.Message(content="🤔 Analyzing the contract and knowledge base...").send()

    try:
        answer = rag_chain.invoke(question)
    except Exception as e:
        await cl.Message(content=f"❌ Error generating answer: {e}").send()
        return

    # -----------------------------------------------------
    # 3) RETRIEVE CLAUSES (Contract-first, KB fallback)
    # -----------------------------------------------------
    try:
        faiss_db = _load_faiss_index()
        retrieved_docs = retrieve_with_pdf_priority(question, faiss_db)
    except Exception as e:
        await cl.Message(
            content=f"⚠️ Answer was generated, but clause retrieval failed: {e}"
        ).send()
        return

    # Separate contract vs KB clauses
    contract_docs = []
    kb_docs = []

    for d in retrieved_docs:
        if d.metadata.get("contract_name") == CURRENT_UPLOADED_CONTRACT:
            contract_docs.append(d)
        else:
            kb_docs.append(d)

    # -----------------------------------------------------
    # 4) Build clause markdown
    # -----------------------------------------------------
    md = []

    if contract_docs:
        md.append("### 📑 Clauses from uploaded contract\n")
        for idx, d in enumerate(contract_docs, start=1):
            md.append(
                f"**Contract Clause {idx}** *(chunk {d.metadata.get('chunk_id', '?')})*:\n\n"
            )
            md.append(d.page_content.strip() + "\n\n---\n\n")

    if kb_docs:
        md.append("### 📚 Clauses from reference knowledge base\n")
        for idx, d in enumerate(kb_docs, start=1):
            md.append(
                f"**KB Clause {idx}** *(chunk {d.metadata.get('chunk_id', '?')})*:\n\n"
            )
            md.append(d.page_content.strip() + "\n\n---\n\n")

    clauses_markdown = "".join(md) if md else "_No clauses retrieved._"

    # -----------------------------------------------------
    # 5) Send answer + retrieved clauses
    # -----------------------------------------------------
    await cl.Message(
        content=answer,
        elements=[
            cl.Text(
                name="Retrieved Clauses",
                content=clauses_markdown,
            )
        ],
    ).send()
