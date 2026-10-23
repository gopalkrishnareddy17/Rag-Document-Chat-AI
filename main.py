import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
import ollama

load_dotenv()
MODEL = os.getenv("CHAT_MODEL")


app = FastAPI(title="RAG Document Chat AI")

class AskRequest(BaseModel):
    question: str
    language :  str

class AskResponse(BaseModel):
    answer: str
    model: str

@app.get("/model")
def get_model():
    return {"model":MODEL}

@app.get("/health")
def health():
    return {"status" : "ok"}

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    response = ollama.chat(
        model= MODEL,
        messages=[
            {"role": "system", "content" :f"You are Pirate. Answer like a pirate. Answer in {req.language}"},
            {"role":"user","conten":req.question},
        ],
    )
    return AskResponse(answer=response["message"]["content"], model=MODEL)