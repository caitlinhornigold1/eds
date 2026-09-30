import json
import os
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Resolve exact absolute paths relative to THIS script (docCleaning/ingest_vectors.py)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Moves UP from docCleaning/ to project root, then INTO edstest/my_vector_db
DB_PATH = os.path.abspath(
    os.path.join(SCRIPT_DIR, "..", "edstest", "my_vector_db")
)

# Resolves docCleaning/output_json
JSON_FOLDER = os.path.abspath(os.path.join(SCRIPT_DIR, "output_json"))

print(f"Targeting Vector DB at: {DB_PATH}")
print(f"Reading JSONs from:     {JSON_FOLDER}\n")

# 2. Connect to ChromaDB using the verified absolute path
chroma_client = chromadb.PersistentClient(path=DB_PATH)

# 3. Get or create collection
collection = chroma_client.get_or_create_collection(name="eds_documents")

# 4. Initialize LangChain's Recursive Text Splitter
# Splits on paragraph breaks ("\n\n"), line breaks ("\n"), spaces, and characters in priority order
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=["\n\n", "\n", " ", ""],
)

if not os.path.exists(JSON_FOLDER):
    print(f"ERROR: Directory '{JSON_FOLDER}' does not exist.")
    exit(1)

# Process all JSON files
for filename in os.listdir(JSON_FOLDER):
    if filename.endswith(".json"):
        file_path = os.path.join(JSON_FOLDER, filename)

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Extract text and metadata
        doc_text = data.get("cleaned_body_text", "")
        doc_metadata = data.get("metadata", {})

        # Ensure metadata values are valid for ChromaDB
        clean_metadata = {
            k: (v if v is not None else "") for k, v in doc_metadata.items()
        }

        # Base ID from filename (e.g., "press_release_1")
        base_doc_id = filename.replace(".json", "")

        if not doc_text.strip():
            continue

        # 5. Split document text using LangChain
        chunks = text_splitter.split_text(doc_text)

        # Create unique IDs for every chunk (e.g., "press_release_1_chunk_0")
        chunk_ids = [f"{base_doc_id}_chunk_{i}" for i in range(len(chunks))]
        chunk_metadatas = [clean_metadata for _ in range(len(chunks))]

        # 6. Add all chunks to ChromaDB
        collection.add(
            documents=chunks, metadatas=chunk_metadatas, ids=chunk_ids
        )

        print(
            f"Ingested into Vector DB: {filename} -> {len(chunks)} chunk(s)"
        )

print("\nVector Database successfully populated!")