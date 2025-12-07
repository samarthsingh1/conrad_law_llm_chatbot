import chainlit as cl
from pathlib import Path

from rag_backend import (
    get_rag_chain,
    process_pdf_and_add_to_vector_db,
)

# Initialize RAG chain
rag_chain = get_rag_chain()


@cl.on_chat_start
async def start_chat():
    await cl.Message(
        content=(
            "**📑 Contract RAG Assistant**\n\n"
            "Upload a contract PDF, then ask questions like:\n"
            "- *What does Clause 5 say?*\n"
            "- *Explain confidentiality in my agreement.*\n"
            "- *What is the governing law section?*\n\n"
            "**Routing Logic:**\n"
            "- If your question mentions **clause**, **section**, **agreement**, **my contract**, etc →\n"
            "  It queries your *USER contract vector DB*.\n"
            "- Otherwise → It queries the **CUAD Legal Knowledge Base**.\n"
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
                    content=f"📄 Upload received: **{file_name}**\nExtracting clauses..."
                ).send()

                # Add extracted clauses to the user's FAISS DB
                await process_pdf_and_add_to_vector_db(pdf_path)

                # Refresh RAG chain to include new contract
                rag_chain = get_rag_chain()

                await cl.Message(
                    content=(
                        f"✅ **{file_name}** processed successfully.\n"
                        "You may now ask questions about your contract."
                    )
                ).send()

        # If user only uploaded PDF and didn't ask anything → stop here
        if not message.content.strip():
            return

    # 2️⃣ Handle the user’s actual question
    question = message.content.strip()
    if not question:
        await cl.Message(content="Please enter a question.").send()
        return

    await cl.Message(content="🔎 Analyzing your question...").send()

    try:
        answer = rag_chain.invoke(question)
    except Exception as e:
        await cl.Message(content=f"❌ Error during RAG processing:\n{e}").send()
        return

    # 3️⃣ Send answer
    await cl.Message(content=answer).send()
