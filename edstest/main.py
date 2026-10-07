import logging
import os
import re
from typing import Any, Dict

import chromadb
from chromadb.errors import ChromaError
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import ollama
from pydantic import BaseModel, Field, field_validator

# 1. Configure Terminal Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("eds_api")

app = FastAPI(title="EDS Environmental RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Dynamic Absolute Pathing for Vector DB
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "my_vector_db"))
COLLECTION_NAME = "eds_documents"

# Maximum distance allowed for a retrieved document chunk to be considered relevant.
# Chunks above this threshold will not be attached as sources.
MAX_SOURCE_DISTANCE = 1.15


def get_db_collection():
    """Helper function to obtain ChromaDB client & collection with error handling."""
    try:
        chroma_client = chromadb.PersistentClient(path=DB_PATH)
        return chroma_client.get_collection(name=COLLECTION_NAME)
    except Exception as e:
        logger.error(f"[DB CONNECTION FAILURE] Path '{DB_PATH}': {str(e)}", exc_info=True)
        raise e


# ------------------------------------------------------------------
# Request & Response Schemas
# ------------------------------------------------------------------
class Question(BaseModel):
    """Input payload model with length constraints and custom sanitization."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="User search query or question",
        examples=["What are the rules for marine protected areas?"],
    )

    @field_validator("question")
    @classmethod
    def validate_and_sanitize_question(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Question cannot be empty or contain only whitespace.")
        cleaned = cleaned.replace("\x00", "")
        return cleaned


def sanitize_response(data: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure no secret tokens leak in response payloads."""
    secret_patterns = [
        r"AIzaSy[A-Za-z0-9_-]{35}",
        r"sk-[A-Za-z0-9]{32,}",
        r"bearer\s+[A-Za-z0-9\-\._~\+\/]+=*",
    ]

    def _clean_str(val: str) -> str:
        for pattern in secret_patterns:
            val = re.sub(pattern, "[REDACTED_SECRET]", val, flags=re.IGNORECASE)
        return val

    if "answer" in data and isinstance(data["answer"], str):
        data["answer"] = _clean_str(data["answer"])

    return data


# ------------------------------------------------------------------
# Exception Handlers
# ------------------------------------------------------------------
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Intercepts validation failures and returns a sanitized 422 response."""
    logger.warning(
        f"[INPUT REJECTED] Path: {request.url.path} | Technical Error: {exc.errors()}"
    )

    first_error = exc.errors()[0]
    error_type = first_error.get("type", "")

    if "string_above_max_length" in error_type or "max_length" in error_type:
        user_detail = "Your question is too long. Please limit your input to 1,000 characters."
    elif "value_error" in error_type or "min_length" in error_type:
        user_detail = "Please enter a valid question before submitting."
    else:
        user_detail = "Invalid request format or parameter types provided."

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": user_detail},
    )


# ------------------------------------------------------------------
# API Endpoints
# ------------------------------------------------------------------
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
        logger.error(f"[HEALTH CHECK FAILED] Database connectivity issue: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The search service is currently undergoing maintenance. Please try again later.",
        )


@app.post("/chat")
def chat(question: Question):
    user_query = question.question

    # --- Step 0: Early Exit for Small Talk / Greetings / Farewells ---
    normalized_query = re.sub(r"[^\w\s]", "", user_query.strip().lower())

    greetings = {"hi", "hello", "hey", "kia ora", "greetings", "good morning", "good afternoon", "good evening"}
    farewells = {"bye", "goodbye", "see ya", "see you", "farewell", "thanks", "thank you", "cheers"}

    if normalized_query in greetings:
        return {
            "answer": "Kia ora! How can I help you with Environmental Defence Society (EDS) publications or policy documents today?",
            "sources": [],
            "low_confidence": False,
            "unanswered": False,
        }

    if normalized_query in farewells:
        return {
            "answer": "Goodbye! Feel free to reach out whenever you have questions about EDS documents or environmental policy.",
            "sources": [],
            "low_confidence": False,
            "unanswered": False,
        }

    # --- Step 1: Query ChromaDB Vector Database ---
    try:
        collection = get_db_collection()
        search_results = collection.query(
            query_texts=[user_query],
            n_results=8,
            include=["documents", "metadatas", "distances"],
        )
    except (ChromaError, Exception) as db_err:
        logger.error(f"[DATABASE QUERY ERROR] Query '{user_query}' failed: {str(db_err)}", exc_info=True)
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

        if document not in source_map:
            source_map[document] = f"Doc {doc_counter}"
            doc_counter += 1

        doc_label = source_map[document]

        formatted_context_blocks.append(
            f"SOURCE [{doc_label}] (Document: {document}, Page {page}, Distance: {dist:.3f}):\n{clean_text}"
        )

    context = "\n\n---\n\n".join(formatted_context_blocks)

    # --- Step 3: Zero-Hallucination System Prompt ---
    system_prompt = f"""You are the EDS information assistant.

Answer the user's question clearly and accurately using the retrieved EDS sources as your primary evidence.

STRICT RULES:
1. ONLY USE PROVIDED CONTEXT: Answer the user question using ONLY facts directly mentioned in the <context> tags below.
2. ABSOLUTE ZERO OUTSIDE KNOWLEDGE: Do NOT use any prior training data, general world knowledge, or outside assumptions. If a fact is not explicitly stated in the context, treat it as entirely unknown.
3. Grounding & Synthesis: Answer the user's question thoroughly by synthesizing all relevant facts, legal principles, and details found in the CONTEXT below. Do NOT introduce outside facts not supported by the context.
4. FORMATTING & READABILITY (CRITICAL):
   - Never output a single block of text.
   - Separate distinct ideas into short paragraphs (2–4 sentences).
   - Use bolding (`**term**`) for headers.
   - INLINE LIST FORMATTING: When defining terms or listing items with descriptions, keep the term, colon, and description on the EXACT SAME LINE using standard bullet points (e.g., "* **Resource Management Plans (RMPs):** Developed by local authorities..."). NEVER place colons or descriptions on a new line beneath a title.
   - Use Markdown bullet points (`*` or `-`) when listing items, ecological features, or legal mechanisms.
5. Unsupported Queries: If the provided context contains NO relevant information to answer the question, state verbatim: "I do not have enough information in my database to answer this."
6. Framing & Depth: Provide a complete, clear, and informative response. When asked "what is" a major legislative or policy instrument, explain its purpose, key frameworks, and legal context as detailed in the documents rather than providing a single basic definition.
7. Don't cite sources inline.
8. Small Talk & Greetings: Keep responses brief (1–2 sentences). Do not explain what you can or cannot do at length unless specifically requested.

CONTEXT:
{context}

USER QUESTION:
{user_query}"""

    # --- Step 4: AI Model Inference ---
    try:
        response = ollama.chat(
            model="llama3.2",
            messages=[{"role": "user", "content": system_prompt}],
            options={"temperature": 0.0},
        )
        answer = response["message"]["content"].strip()
    except Exception as model_err:
        logger.error(f"[AI MODEL INFERENCE FAILURE] Model execution error for query '{user_query}': {str(model_err)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="We ran into an issue generating your response. Please try asking your question again.",
        )

    # --- Step 5: Detect Unsupported / Low-Confidence Refusals ---
    answer_clean = answer.lower()

    unsupported_signals = [
        "i do not have enough information in my database to answer",
        "i do not have enough information",
        "i don't have enough information",
        "not enough information in my database",
        "does not contain information",
        "insufficient information",
        "context provided does not contain",
        "clarify what you would like to know",
        "is there anything else i can help you with before you go",
        "it seems you're saying goodbye",
    ]

    fully_unsupported = any(signal in answer_clean for signal in unsupported_signals)
    low_confidence = fully_unsupported

    # --- Step 6: Extract Unique Sources (With Distance Filter & Refusal Guard) ---
    sources = []
    seen_sources = set()

    if not fully_unsupported:
        for rank_idx, (meta, dist) in enumerate(zip(retrieved_metas, retrieved_distances), start=1):
            # Skip chunks with high vector distance (i.e. low relevance)
            if dist > MAX_SOURCE_DISTANCE:
                continue

            document = meta.get("document", "Unknown Document")
            page = meta.get("page")
            url = meta.get("url")

            if not url and document != "Unknown Document":
                clean_filename = document.replace(" ", "_")
                url = f"https://www.eds.org.nz/publications/{clean_filename}"

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

    return sanitize_response(response_payload)