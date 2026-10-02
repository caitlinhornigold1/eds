import logging
import os
import re
from typing import Any, Dict

import chromadb
from chromadb.errors import ChromaError
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import ollama
from pydantic import BaseModel

# 1. Configure Terminal Logging (Outputs full technical stack traces to server console)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("eds_api")

app = FastAPI(title="EDS Environmental RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Dynamic Absolute Pathing for Vector DB
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "my_vector_db"))
COLLECTION_NAME = "eds_documents"


def get_db_collection():
    """Helper function to obtain ChromaDB client & collection with error handling."""
    try:
        chroma_client = chromadb.PersistentClient(path=DB_PATH)
        return chroma_client.get_collection(name=COLLECTION_NAME)
    except Exception as e:
        # CONSOLE: Log internal file path and DB connection details
        logger.error(f"[DB CONNECTION FAILURE] Path '{DB_PATH}': {str(e)}", exc_info=True)
        raise e


class Question(BaseModel):
    question: str


def sanitize_response(data: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure no secret tokens, private paths, or raw authorization strings leak in response payloads."""
    secret_patterns = [
        r"AIzaSy[A-Za-z0-9_-]{35}",  # GCP / Gemini API Keys
        r"sk-[A-Za-z0-9]{32,}",      # OpenAI / generic API secret keys
        r"bearer\s+[A-Za-z0-9\-\._~\+\/]+=*",  # Bearer tokens
    ]

    def _clean_str(val: str) -> str:
        for pattern in secret_patterns:
            val = re.sub(pattern, "[REDACTED_SECRET]", val, flags=re.IGNORECASE)
        return val

    if "answer" in data and isinstance(data["answer"], str):
        data["answer"] = _clean_str(data["answer"])

    return data


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """Health endpoint to monitor application uptime and vector database status."""
    try:
        collection = get_db_collection()
        doc_count = collection.count()
        return {
            "status": "healthy",
            "database": "connected",
            "collection": COLLECTION_NAME,
            "document_chunk_count": doc_count,
        }
    except Exception as e:
        # CONSOLE: Log technical error details
        logger.error(f"[HEALTH CHECK FAILED] Database connectivity issue: {str(e)}", exc_info=True)
        # WEBSITE: Return generic non-technical detail
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The search service is currently undergoing maintenance. Please try again later.",
        )


@app.post("/chat")
def chat(question: Question):
    user_query = question.question

    # --- Step 1: Query ChromaDB Vector Database ---
    try:
        collection = get_db_collection()
        search_results = collection.query(
            query_texts=[user_query],
            n_results=8,
            include=["documents", "metadatas", "distances"],
        )
    except (ChromaError, Exception) as db_err:
        # CONSOLE: Detailed technical error & stack trace printed in server terminal
        logger.error(f"[DATABASE QUERY ERROR] Query '{user_query}' failed: {str(db_err)}", exc_info=True)

        # WEBSITE: Non-technical message returned to the user in the chatbot UI
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Our search service is temporarily unavailable. Please try your question again in a few moments.",
        )

    retrieved_docs = search_results["documents"][0] if search_results.get("documents") else []
    retrieved_metas = search_results["metadatas"][0] if search_results.get("metadatas") else []
    retrieved_distances = search_results["distances"][0] if search_results.get("distances") else []

    # --- Step 2: Deduplicate Chunks & Map Sources in Order of Relevance ---
    seen_texts = set()
    source_map = {}
    formatted_context_blocks = []
    doc_counter = 1

    for doc, meta, dist in zip(retrieved_docs, retrieved_metas, retrieved_distances):
        clean_text = doc.strip()

        if clean_text in seen_texts:
            continue
        seen_texts.add(clean_text)

        document = meta.get("document", "Unknown Document")
        page = meta.get("page", "N/A")

        # First occurrence of a document establishes its rank label (Doc 1 = most relevant)
        if document not in source_map:
            source_map[document] = f"Doc {doc_counter}"
            doc_counter += 1

        doc_label = source_map[document]

        formatted_context_blocks.append(
            f"SOURCE [{doc_label}] (Document: {document}, Page {page}, Distance: {dist:.3f}):\n{clean_text}"
        )

    context = "\n\n---\n\n".join(formatted_context_blocks)

    # --- Step 3: Zero-Hallucination System Prompt ---
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

    # --- Step 4: AI Model Inference ---
    try:
        response = ollama.chat(
            model="llama3.2",
            messages=[{"role": "user", "content": system_prompt}],
            options={"temperature": 0.0},
        )
        answer = response["message"]["content"]
    except Exception as model_err:
        # CONSOLE: Detailed technical error & stack trace printed in server terminal
        logger.error(f"[AI MODEL INFERENCE FAILURE] Model execution error for query '{user_query}': {str(model_err)}", exc_info=True)

        # WEBSITE: Non-technical message returned to the user in the chatbot UI
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="We ran into an issue generating your response. Please try asking your question again.",
        )

    insufficient_phrase = "i do not have enough information in my database to answer"
    low_confidence = insufficient_phrase in answer.lower()
    fully_unsupported = answer.strip().lower().startswith(insufficient_phrase)

    # --- Step 5: Extract Unique Sources Preserving Exact Relevance Order ---
    sources = []
    seen_sources = set()

    if not fully_unsupported:
        for rank_idx, (meta, dist) in enumerate(zip(retrieved_metas, retrieved_distances), start=1):
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
                "rank": rank_idx,
                "distance": round(dist, 4),
            }

            if page is not None:
                source["page"] = page

            if url:
                source["url"] = url

            sources.append(source)

    response_payload = {
        "answer": answer,
        "sources": sources,
        "low_confidence": low_confidence,
        "unanswered": fully_unsupported,
    }

    # Clean and return safe payload
    return sanitize_response(response_payload)