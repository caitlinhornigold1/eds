import os
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

# 1. Dynamic Absolute Pathing for Vector DB
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "my_vector_db"))

chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection = chroma_client.get_collection(name="eds_documents")


class Question(BaseModel):
    question: str


@app.post("/chat")
def chat(question: Question):
    user_query = question.question

    # 1. Retrieve top matching document chunks (increased to 8 for multi-source coverage)
    search_results = collection.query(
        query_texts=[user_query],
        n_results=8,
    )

    retrieved_docs = search_results["documents"][0]
    retrieved_metas = search_results["metadatas"][0]

    # 2. Deduplicate chunks & Map Sources to [Doc X] labels
    seen_texts = set()
    source_map = {}
    formatted_context_blocks = []
    doc_counter = 1

    for doc, meta in zip(retrieved_docs, retrieved_metas):
        clean_text = doc.strip()

        # Skip duplicate text chunks
        if clean_text in seen_texts:
            continue
        seen_texts.add(clean_text)

        document = meta.get("document", "Unknown Document")
        page = meta.get("page", "N/A")

        # Assign a consistent label like [Doc 1], [Doc 2] per unique file
        if document not in source_map:
            source_map[document] = f"Doc {doc_counter}"
            doc_counter += 1

        doc_label = source_map[document]

        formatted_context_blocks.append(
            f"SOURCE [{doc_label}] (Document: {document}, Page {page}):\n{clean_text}"
        )

    context = "\n\n---\n\n".join(formatted_context_blocks)

    # 3. Strict Zero-Hallucination System Prompt with Citation Rules
    system_prompt = f"""
    You are a strict data-extraction assistant for the Environmental Defence Society (EDS).
    
    STRICT RULES:
    1. Answer the question using ONLY the verbatim facts in the CONTEXT below.
    2. Every factual statement or claim MUST be cited inline using its corresponding source tag, e.g., [Doc 1] or [Doc 2].
    3. DO NOT infer, extrapolate, or assume any politician's or party's stance unless explicitly stated in the context.
    4. If the provided context does not contain enough information to answer fully, state what is known from the context without adding meta-commentary about your database.
    5. If the provided context contains NO relevant information, state: "I do not have enough information in my database to answer this."

    CONTEXT:
    {context}

    USER QUESTION:
    {user_query}
    """

    # 4. Call Ollama
    response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": system_prompt}],
        options={"temperature": 0.0},
    )

    answer = response["message"]["content"]

    insufficient_phrase = "i do not have enough information in my database to answer"
    low_confidence = insufficient_phrase in answer.lower()
    fully_unsupported = answer.strip().lower().startswith(insufficient_phrase)

    # 5. Extract Unique Sources for Response Payload (Processed outside the loop!)
    sources = []
    seen_sources = set()

    if not fully_unsupported:
        for meta in retrieved_metas:
            document = meta.get("document", "Unknown Document")
            page = meta.get("page")
            url = meta.get("url")

            source_key = (document, page, url)

            if source_key in seen_sources:
                continue

            seen_sources.add(source_key)

            doc_label = source_map.get(document, "Doc")

            source = {
                "label": doc_label,
                "document": document,
            }

            if page is not None:
                source["page"] = page

            if url:
                source["url"] = url

            sources.append(source)

    return {
        "answer": answer,
        "sources": sources,
        "low_confidence": low_confidence,
        "unanswered": fully_unsupported,
    }