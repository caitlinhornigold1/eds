import json
import os
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.abspath(
    os.path.join(SCRIPT_DIR, "..", "edstest", "my_vector_db")
)
JSON_FOLDER = os.path.abspath(os.path.join(SCRIPT_DIR, "output_json"))

print(f"Targeting Vector DB at: {DB_PATH}")
print(f"Reading JSONs from:     {JSON_FOLDER}\n")

chroma_client = chromadb.PersistentClient(path=DB_PATH)

# --- CLEAN INDEX RESET ---
# 1. Safely delete the collection if it already exists
try:
    chroma_client.delete_collection(name="eds_documents")
    print("Deleted old 'eds_documents' collection to prevent duplicates.")
except Exception:
    print("No existing 'eds_documents' collection found. Creating new one.")

# 2. Create a fresh collection
collection = chroma_client.create_collection(name="eds_documents")
# -------------------------

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

        doc_text = data.get("cleaned_body_text", "")
        doc_metadata = data.get("metadata", {})

        clean_metadata = {
            k: (v if v is not None else "") for k, v in doc_metadata.items()
        }

        base_doc_id = filename.replace(".json", "")

        if not doc_text.strip():
            continue

        chunks = text_splitter.split_text(doc_text)
        chunk_ids = [f"{base_doc_id}_chunk_{i}" for i in range(len(chunks))]
        chunk_metadatas = [clean_metadata for _ in range(len(chunks))]

        collection.add(
            documents=chunks, metadatas=chunk_metadatas, ids=chunk_ids
        )

        print(
            f"Ingested into Vector DB: {filename} -> {len(chunks)} chunk(s)"
        )

print("\nVector Database successfully wiped and re-populated!")