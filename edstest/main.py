import chromadb

from fastapi import FastAPI

from fastapi.middleware.cors import CORSMiddleware

import ollama

from pydantic import BaseModel


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# 1. Connect to the ChromaDB vector database

chroma_client = chromadb.PersistentClient(path="./my_vector_db")

collection = chroma_client.get_collection(name="eds_documents")


class Question(BaseModel):

    question: str


@app.post("/chat")
def chat(question: Question):
    user_query = question.question

    # 1. Retrieve top matching document chunks
    search_results = collection.query(
        query_texts=[user_query],
        n_results=2,
    )

    retrieved_docs = search_results["documents"][0]
    retrieved_metas = search_results["metadatas"][0]

    context = ""
    for idx, (doc, meta) in enumerate(
        zip(retrieved_docs, retrieved_metas), start=1
    ):
        source_title = meta.get("document", "Unknown Document")
        context += f"\n--- DOCUMENT SOURCE {idx}: {source_title} ---\n{doc}\n"

    # 2. Strict Zero-Hallucination System Prompt
    system_prompt = f"""
    You are a strict data-extraction assistant for the Environmental Defence Society (EDS).
    
    STRICT RULES:
    1. Answer the question using ONLY the verbatim facts in the CONTEXT below.
    2. DO NOT infer, extrapolate, or assume any politician's or party's stance unless it is EXPLICITLY stated in the context.
    3. If a political party or person is NOT mentioned regarding a specific topic, DO NOT include them in your summary.
    4. If the provided context does not contain enough information to answer, state: "I do not have enough information in my database to answer this."

    CONTEXT:
    {context}

    USER QUESTION:
    {user_query}
    """

    # 3. Call Ollama with temperature=0.0 to prevent creative leaps
    response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": system_prompt}],
        options={
            "temperature": 0.0  # Forces deterministic, non-creative extraction
        },
    )

    return {"answer": response["message"]["content"]}