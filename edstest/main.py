from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import ollama

app = FastAPI()

# Allow the website to communicate with our Python backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Question(BaseModel):
    question: str


@app.post("/chat")
def chat(question: Question):

    response = ollama.chat(
        model="llama3.2",
        messages=[
            {
                "role": "user",
                "content": question.question
            }
        ]
    )

    return {
        "answer": response["message"]["content"]
    }