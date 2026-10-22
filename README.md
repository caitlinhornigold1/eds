# eds

to run chatbot:
1. source venv/bin/activate (mac only)
2. cd docCleaning
3. python ingest_vectors.py
4. python query_db.py
5. cd ..
6. cd edstest
7. uvicorn main:app --reload   