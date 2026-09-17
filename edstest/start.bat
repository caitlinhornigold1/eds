@echo off

call venv\Scripts\activate

echo Starting EDS Chatbot backend...

uvicorn main:app --reload

pause