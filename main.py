from fastapi import FastAPI, UploadFile, File, HTTPException
from dotenv import load_dotenv
from pydantic import BaseModel
from rag import ingest_pdf, collection
import shutil
import ollama
import os


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
            {"role": "system", "content" :f"Hello i am assistant how can i help you. Answer in {req.language}"},
            # {"role": "system", "content" :"You are Pirate. Answer likea a pirate"},
            {"role":"user","conten":req.question},
        ],
    )
    return AskResponse(answer=response["message"]["content"], model=MODEL)

@app.post("/ingest")
def ingest(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only pdf files are supported")
    path = os.path.join("data", file.filename)
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    count = ingest_pdf(path, file.filename)
    return {"file": file.filename, "chunks":count}

@app.get("/status")
def stats():
    return {"chunks_stored":collection.count()}