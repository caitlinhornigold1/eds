from pathlib import Path

import chromadb
import ollama

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------
# FastAPI
# ---------------------------------------------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# ChromaDB
# ---------------------------------------------------------

chroma_client = chromadb.PersistentClient(
    path=str(BASE_DIR / "my_vector_db")
)

collection = chroma_client.get_collection(
    name="eds_documents"
)


# ---------------------------------------------------------
# Request model
# ---------------------------------------------------------

class Question(BaseModel):
    question: str


# ---------------------------------------------------------
# Chat endpoint
# ---------------------------------------------------------

@app.post("/chat")
def chat(question: Question):

    user_query = question.question.strip()

    if not user_query:
        raise HTTPException(
            status_code=400,
            detail="A question is required."
        )


    # -----------------------------------------------------
    # 1. Retrieve relevant document chunks
    # -----------------------------------------------------

    search_results = collection.query(
        query_texts=[user_query],
        n_results=1,
    )

    retrieved_docs = (
        search_results.get("documents", [[]])[0] or []
    )

    retrieved_metas = (
        search_results.get("metadatas", [[]])[0] or []
    )


    # -----------------------------------------------------
    # 2. Build context
    # -----------------------------------------------------

    context = ""

    for idx, (doc, meta) in enumerate(
        zip(retrieved_docs, retrieved_metas),
        start=1
    ):

        source_title = meta.get(
            "document",
            "Unknown Document"
        )

        context += (
            f"\n--- DOCUMENT SOURCE {idx}: "
            f"{source_title} ---\n"
            f"{doc}\n"
        )


    # -----------------------------------------------------
    # 3. Strict grounding prompt
    # -----------------------------------------------------

    system_prompt = f"""
You are a strict data-extraction assistant for the
Environmental Defence Society (EDS).

STRICT RULES:

1. Answer the user's question using ONLY facts contained
   in the CONTEXT below.

2. Do not infer, extrapolate, assume, or invent facts that
   are not explicitly supported by the context.

3. Do not assume a politician's, political party's,
   organisation's, or person's position unless it is
   explicitly stated in the context.

4. If a political party or person is not mentioned in
   relation to a specific topic, do not include them in
   the answer.

5. If the context contains enough information to answer
   only part of the question, answer only the supported
   part.

6. If the context contains no relevant information that
   can answer the question, respond exactly with:

   I do not have enough information in my database to answer this.

7. Keep the response clear and understandable for a
   general user.

CONTEXT:
{context}

USER QUESTION:
{user_query}
"""


    # -----------------------------------------------------
    # 4. Generate answer using Ollama
    # -----------------------------------------------------

    try:

        response = ollama.chat(
            model="llama3.2",

            messages=[
                {
                    "role": "user",
                    "content": system_prompt
                }
            ],

            options={
                "temperature": 0.0
            }
        )

        answer = (
            response["message"]["content"]
            or ""
        ).strip()

    except Exception as error:

        print(
            f"Ollama error: {error}"
        )

        raise HTTPException(
            status_code=502,
            detail="The AI service is currently unavailable."
        )


    if not answer:

        answer = (
            "I do not have enough information "
            "in my database to answer this."
        )


    # -----------------------------------------------------
    # 5. Confidence / unanswered status
    # -----------------------------------------------------

    insufficient_phrase = (
        "i do not have enough information "
        "in my database to answer"
    )

    low_confidence = (
        insufficient_phrase
        in answer.lower()
    )

    fully_unsupported = (
        answer
        .strip()
        .lower()
        .startswith(insufficient_phrase)
    )


    # -----------------------------------------------------
    # 6. Prepare sources
    # -----------------------------------------------------

    sources = []

    seen_sources = set()

    for meta in retrieved_metas:

        document = meta.get(
            "document",
            "Unknown Document"
        )

        page = meta.get("page")
        url = meta.get("url")

        source_key = (
            document,
            page,
            url
        )

        if source_key in seen_sources:
            continue

        seen_sources.add(
            source_key
        )

        source = {
            "document": document
        }

        if page is not None:
            source["page"] = page

        if url:
            source["url"] = url

        sources.append(
            source
        )


    # Do not show unrelated sources for completely
    # unanswered questions
    if fully_unsupported:
        sources = []


    # -----------------------------------------------------
    # 7. Return response to frontend
    # -----------------------------------------------------

    return {
        "answer": answer,
        "sources": sources,
        "low_confidence": low_confidence,
        "unanswered": fully_unsupported
    }