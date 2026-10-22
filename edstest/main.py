import logging
import os
import re
from typing import Any, Dict

import chromadb
import ollama

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger("eds_api")


# ---------------------------------------------------------
# FastAPI
# ---------------------------------------------------------

app = FastAPI(
    title="EDS Environmental RAG API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# ChromaDB
# ---------------------------------------------------------

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DB_PATH = os.path.abspath(
    os.path.join(
        SCRIPT_DIR,
        "my_vector_db"
    )
)

COLLECTION_NAME = "eds_documents"


def get_db_collection():
    """
    Connect to the ChromaDB collection.
    """

    try:
        chroma_client = chromadb.PersistentClient(
            path=DB_PATH
        )

        return chroma_client.get_collection(
            name=COLLECTION_NAME
        )

    except Exception as error:

        logger.error(
            f"[DB CONNECTION FAILURE] "
            f"Path '{DB_PATH}': {error}",
            exc_info=True
        )

        raise


# ---------------------------------------------------------
# Request model
# ---------------------------------------------------------

class Question(BaseModel):
    """
    Input payload for chatbot questions.
    """

    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="User search query or question",
        examples=[
            "What are the rules for marine protected areas?"
        ],
    )

    @field_validator("question")
    @classmethod
    def validate_and_sanitize_question(
        cls,
        value: str
    ) -> str:

        cleaned = value.strip()

        if not cleaned:
            raise ValueError(
                "Question cannot be empty."
            )

        # Remove null bytes
        cleaned = cleaned.replace(
            "\x00",
            ""
        )

        return cleaned


# ---------------------------------------------------------
# Response sanitisation
# ---------------------------------------------------------

def sanitize_response(
    data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Prevent secrets from accidentally appearing
    in chatbot responses.
    """

    secret_patterns = [
        r"AIza[0-9A-Za-z_-]{20,}",
        r"sk-[A-Za-z0-9]{20,}",
        r"bearer\s+[A-Za-z0-9\-._~+/]+=*",
    ]

    def clean_string(
        value: str
    ) -> str:

        for pattern in secret_patterns:

            value = re.sub(
                pattern,
                "[REDACTED_SECRET]",
                value,
                flags=re.IGNORECASE
            )

        return value

    if (
        "answer" in data
        and isinstance(
            data["answer"],
            str
        )
    ):
        data["answer"] = clean_string(
            data["answer"]
        )

    return data


# ---------------------------------------------------------
# Validation error handling
# ---------------------------------------------------------

@app.exception_handler(
    RequestValidationError
)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError
):

    logger.warning(
        f"[INPUT REJECTED] "
        f"Path: {request.url.path} | "
        f"Technical Error: {exc.errors()}"
    )

    first_error = exc.errors()[0]

    error_type = first_error.get(
        "type",
        ""
    )

    if (
        "string_above_max_length"
        in error_type
        or "max_length"
        in error_type
    ):

        user_detail = (
            "Your question is too long. "
            "Please limit your input to "
            "1,000 characters."
        )

    elif (
        "value_error" in error_type
        or "min_length" in error_type
    ):

        user_detail = (
            "Please enter a valid question "
            "before submitting."
        )

    else:

        user_detail = (
            "Invalid request format."
        )

    return JSONResponse(
        status_code=(
            status.HTTP_422_UNPROCESSABLE_ENTITY
        ),
        content={
            "detail": user_detail
        },
    )


# ---------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------

@app.get(
    "/health",
    status_code=status.HTTP_200_OK
)
def health_check():
    """
    Check whether the API and vector database
    are available.
    """

    try:

        collection = get_db_collection()

        document_count = (
            collection.count()
        )

        return {
            "status": "healthy",
            "database": "connected",
            "collection": COLLECTION_NAME,
            "document_chunk_count":
                document_count,
        }

    except Exception as error:

        logger.error(
            f"[HEALTH CHECK FAILED] "
            f"{error}",
            exc_info=True
        )

        raise HTTPException(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            detail=(
                "The search service is currently "
                "undergoing maintenance. "
                "Please try again later."
            ),
        )


# ---------------------------------------------------------
# Chat endpoint
# ---------------------------------------------------------

@app.post("/chat")
def chat(
    question: Question
):

    user_query = question.question


    # -----------------------------------------------------
    # 1. Query ChromaDB
    # -----------------------------------------------------

    try:

        collection = (
            get_db_collection()
        )

        search_results = (
            collection.query(
                query_texts=[
                    user_query
                ],
                n_results=8,
                include=[
                    "documents",
                    "metadatas",
                    "distances"
                ],
            )
        )

    except Exception as db_error:

        logger.error(
            f"[DATABASE QUERY ERROR] "
            f"Query '{user_query}' failed: "
            f"{db_error}",
            exc_info=True
        )

        raise HTTPException(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            detail=(
                "Our search service is temporarily "
                "unavailable. Please try your "
                "question again in a few moments."
            ),
        )


    retrieved_docs = (
        search_results["documents"][0]
        if search_results.get(
            "documents"
        )
        else []
    )

    retrieved_metas = (
        search_results["metadatas"][0]
        if search_results.get(
            "metadatas"
        )
        else []
    )

    retrieved_distances = (
        search_results["distances"][0]
        if search_results.get(
            "distances"
        )
        else []
    )


    # -----------------------------------------------------
    # 2. Deduplicate chunks and prepare context
    # -----------------------------------------------------

    seen_texts = set()

    source_map = {}

    formatted_context_blocks = []

    document_counter = 1


    for doc, meta, distance in zip(
        retrieved_docs,
        retrieved_metas,
        retrieved_distances
    ):

        clean_text = doc.strip()

        if clean_text in seen_texts:
            continue

        seen_texts.add(
            clean_text
        )

        document = meta.get(
            "document",
            "Unknown Document"
        )

        page = meta.get(
            "page",
            "N/A"
        )

        if document not in source_map:

            source_map[document] = (
                f"Doc {document_counter}"
            )

            document_counter += 1


        document_label = (
            source_map[document]
        )


        formatted_context_blocks.append(
            f"SOURCE [{document_label}] "
            f"(Document: {document}, "
            f"Page {page}, "
            f"Distance: {distance:.3f}):\n"
            f"{clean_text}"
        )


    context = (
        "\n\n---\n\n".join(
            formatted_context_blocks
        )
    )


    # -----------------------------------------------------
    # 3. Strict grounding prompt
    # -----------------------------------------------------

    system_prompt = f"""
You are the EDS information assistant.

Answer the user's question clearly and accurately
using ONLY the retrieved EDS sources below.

IMPORTANT RULES:

1. Grounding:
   Answer using only facts supported by the CONTEXT.
   Do not introduce outside information.

2. Synthesis:
   You may combine relevant facts from multiple EDS
   documents when they directly support the answer.

3. Accuracy:
   Do not infer, assume, extrapolate, or invent
   information that is not explicitly supported.

4. Entity distinctions:
   Do not confuse a species, organisation, policy,
   commercial activity, person, or political party
   with another type of entity.

5. Partial information:
   If the context only supports part of the user's
   question, answer only the supported part.

6. Unsupported questions:
   If the context contains NO relevant information,
   respond exactly with:

   I do not have enough information in my database to answer this.

7. Response style:
   Give a clear and understandable answer for a
   general user.

8. Sources:
   Do not include source citations inside the answer.
   Sources will be displayed separately by the
   chatbot interface.

CONTEXT:

{context}

USER QUESTION:

{user_query}
"""


    # -----------------------------------------------------
    # 4. Ollama inference
    # -----------------------------------------------------

    try:

        response = ollama.chat(
            model="llama3.2",
            messages=[
                {
                    "role": "user",
                    "content":
                        system_prompt
                }
            ],
            options={
                "temperature": 0.0
            },
        )

        answer = (
            response["message"][
                "content"
            ]
            or ""
        ).strip()

    except Exception as model_error:

        logger.error(
            f"[AI MODEL INFERENCE FAILURE] "
            f"Query '{user_query}' failed: "
            f"{model_error}",
            exc_info=True
        )

        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=(
                "We ran into an issue generating "
                "your response. Please try asking "
                "your question again."
            ),
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
        .startswith(
            insufficient_phrase
        )
    )


    # -----------------------------------------------------
    # 6. Prepare sources
    # -----------------------------------------------------

    sources = []

    seen_sources = set()


    if not fully_unsupported:

        for rank_index, (
            meta,
            distance
        ) in enumerate(
            zip(
                retrieved_metas,
                retrieved_distances
            ),
            start=1
        ):

            document = meta.get(
                "document",
                "Unknown Document"
            )

            page = meta.get(
                "page"
            )

            url = meta.get(
                "url"
            )


            source_key = (
                document,
                page,
                url
            )


            if (
                source_key
                in seen_sources
            ):
                continue


            seen_sources.add(
                source_key
            )


            document_label = (
                source_map.get(
                    document,
                    "Doc"
                )
            )


            source = {
                "label":
                    document_label,
                "document":
                    document,
                "rank":
                    rank_index,
                "distance":
                    round(
                        distance,
                        4
                    ),
            }


            if page is not None:

                source["page"] = (
                    page
                )


            # Only use real URLs that exist
            # in the document metadata.
            if url:

                source["url"] = (
                    url
                )


            sources.append(
                source
            )


    # -----------------------------------------------------
    # 7. Return response
    # -----------------------------------------------------

    response_payload = {
        "answer":
            answer,
        "sources":
            sources,
        "low_confidence":
            low_confidence,
        "unanswered":
            fully_unsupported,
    }


    return sanitize_response(
        response_payload
    )