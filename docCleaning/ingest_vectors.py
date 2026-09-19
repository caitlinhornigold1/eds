import json
import os
import chromadb

# 1. Initialize a local vector database stored in a folder called 'my_vector_db'
chroma_client = chromadb.PersistentClient(path="../edtest/my_vector_db")

# 2. Get or create a collection (like a table in SQL)
collection = chroma_client.get_or_create_collection(name="eds_documents")

# Path where your JSON files live
JSON_FOLDER = "output_json"

# Process all JSON files
for filename in os.listdir(JSON_FOLDER):
    if filename.endswith(".json"):
        file_path = os.path.join(JSON_FOLDER, filename)

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Extract text and metadata from your JSON structure
        doc_text = data.get("cleaned_body_text", "")
        doc_metadata = data.get("metadata", {})

        # Ensure metadata values are valid (Chroma prefers strings, ints, floats, booleans)
        clean_metadata = {
            k: (v if v is not None else "") for k, v in doc_metadata.items()
        }

        # Unique identifier for the vector record
        doc_id = filename.replace(".json", "")

        # 3. Add to ChromaDB
        # Chroma automatically handles vectorizing doc_text using its built-in model
        collection.add(
            documents=[doc_text], metadatas=[clean_metadata], ids=[doc_id]
        )

        print(f"Ingested into Vector DB: {filename}")

print("\nVector Database successfully populated!")