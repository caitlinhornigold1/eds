import json
import chromadb
import os

DB_PATH = os.path.abspath("./edstest/my_vector_db")  # Update path as needed
client = chromadb.PersistentClient(path=DB_PATH)

# 1. Collection of test questions + expected source documents
GOLDEN_DATASET = [
    {
        "question": "What are the rules regarding marine protected areas?",
        "expected_doc": "Foreshore and Seabed Act 2004",
    },
    {
        "question": "What is the sustainable catch limit for Tarakihi?",
        "expected_doc": "Tarakihi",
    },
]

# Initialize ChromaDB client
client = chromadb.PersistentClient(path="edstest/my_vector_db")
collection = client.get_collection(name="eds_documents")


def evaluate_retrieval(top_k=3):
    incorrect_retrievals = []

    for item in GOLDEN_DATASET:
        query = item["question"]
        expected = item["expected_doc"]

        # Run vector search
        results = collection.query(query_texts=[query], n_results=top_k)

        # Extract retrieved document names from metadata
        retrieved_metas = results["metadatas"][0]
        retrieved_docs = [meta.get("document") for meta in retrieved_metas]

        # 3. Compare against expected information
        hit = any(expected.lower() in doc.lower() for doc in retrieved_docs)

        # 4. Record incorrect retrievals
        if not hit:
            incorrect_retrievals.append(
                {
                    "question": query,
                    "expected_doc": expected,
                    "retrieved_docs": retrieved_docs,
                }
            )

    print(
        f"Evaluation Complete! Accuracy: {len(GOLDEN_DATASET) - len(incorrect_retrievals)}/{len(GOLDEN_DATASET)}"
    )

    if incorrect_retrievals:
        print("\n--- Missed Retrievals ---")
        for failure in incorrect_retrievals:
            print(f"Query: {failure['question']}")
            print(f"Expected: {failure['expected_doc']}")
            print(f"Got: {failure['retrieved_docs']}\n")


if __name__ == "__main__":
    evaluate_retrieval()