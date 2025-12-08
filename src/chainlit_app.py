import chainlit as cl
from pathlib import Path


from rag_backend import (
    get_rag_chain,
    process_pdf_and_add_to_vector_db,
    format_top_k_clauses,
    retrieve_docs
)

# Initialize RAG chain
rag_chain = get_rag_chain()


@cl.on_chat_start
async def start_chat():
    await cl.Message(
        content=(
            "*📑 Contract RAG Assistant*\n\n"
            "Upload a contract PDF, then ask questions like:\n"
            "- What does Clause 5 say?\n"
            "- Explain confidentiality in my agreement.\n"
            "- What is the governing law section?\n\n"
            "*Routing Logic:*\n"
            "- If your question mentions *clause, **section, **agreement, **my contract*, etc →\n"
            "  It queries your USER contract vector DB.\n"
            "- Otherwise → It queries the *CUAD Legal Knowledge Base*.\n"
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    global rag_chain

    # 1️⃣ Handle PDF uploads
    if message.elements:
        for element in message.elements:
            if isinstance(element, cl.File) and element.path.lower().endswith(".pdf"):
                pdf_path = element.path
                file_name = element.name or Path(pdf_path).name

                await cl.Message(
                    content=f"📄 Upload received: *{file_name}*\nExtracting clauses..."
                ).send()

                await process_pdf_and_add_to_vector_db(pdf_path)

                rag_chain = get_rag_chain()

                await cl.Message(
                    content=(
                        f"✅ *{file_name}* processed successfully.\n"
                        "You may now ask questions about your contract."
                    )
                ).send()

        if not message.content.strip():
            return

    # 2️⃣ Handle actual user query
    question = message.content.strip()
    if not question:
        await cl.Message(content="Please enter a question.").send()
        return

    await cl.Message(content="🔎 Analyzing your question...").send()

    # -------------------------------------------------------------
    # NEW: Retrieve clauses BEFORE answering
    # -------------------------------------------------------------
    try:
        user_docs, kb_docs = retrieve_docs(question, k=8)
        retrieved_docs = user_docs + kb_docs

        top_k_markdown = format_top_k_clauses(retrieved_docs, k=5)

        await cl.Message(
            content=(
                "### 📌 Retrieved Clauses Used for Answer\n"
                "(Development Mode)\n\n"
                + top_k_markdown
            )
        ).send()

    except Exception as e:
        await cl.Message(content=f"⚠️ Retrieval error:\n{e}").send()

    # -------------------------------------------------------------
    # Generate the final LLM answer
    # -------------------------------------------------------------
    try:
        answer = rag_chain.invoke(question)
    except Exception as e:
        await cl.Message(content=f"❌ Error during RAG processing:\n{e}").send()
        return

    # 3️⃣ Send final answer
    await cl.Message(
        content=(
            "### 🤖 Final Answer\n\n"
            f"{answer}"
        )
    ).send()
