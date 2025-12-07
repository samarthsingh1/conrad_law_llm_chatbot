import chainlit as cl
from rag_backend import get_rag_chain

# Load chain on startup
rag_chain = get_rag_chain()

@cl.on_chat_start
async def start_chat():
    await cl.Message("""
👋 **Contract RAG Assistant Ready**

I can answer questions strictly based on your FAISS-indexed contract clauses.
Ask me anything!
    """).send()

@cl.on_message
async def on_message(msg: cl.Message):
    query = msg.content
    answer = rag_chain.invoke(query)

    await cl.Message(answer).send()
