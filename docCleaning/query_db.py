import json
import chromadb

# 1. Connect to the existing local database
chroma_client = chromadb.PersistentClient(path="./my_vector_db")
collection = chroma_client.get_collection(name="eds_documents")

# 2. Define your search term
search_query = "What is the policy on electricity and LNG?"

# 3. Query the collection for the Top 2 matches
results = collection.query(query_texts=[search_query], n_results=2)

# --- FORMATTED OUTPUT ---
print(f"\n Search Query: '{search_query}'\n" + "=" * 50)

# Unpack results lists
documents = results["documents"][0]
metadatas = results["metadatas"][0]
distances = results["distances"][0]

for idx, (doc, meta, dist) in enumerate(
    zip(documents, metadatas, distances), start=1
):
    print(f"\n--- MATCH #{idx} (Similarity Distance: {dist:.4f}) ---")
    print(f"Document Title : {meta.get('document', 'N/A')}")
    print(f"Organisation   : {meta.get('organisation', 'N/A')}")
    print(f"Date           : {meta.get('date', 'N/A')}")
    print(f"Source URL     : {meta.get('url', 'N/A')}")
    print("\nMatched Text:")
    print(
        doc[:300] + "..." if len(doc) > 300 else doc
    )  # Preview first 300 chars
    print("=" * 50)