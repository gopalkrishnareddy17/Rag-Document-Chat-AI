from pypdf import PdfReader
import ollama
import chromadb
import os;

CHUNK_SIZE = 100
CHUNK_OVERLAP = 0
EMBED_MODEL = "nomic-embed-text"
BATCH_SIZE = 32
TOP_K=5

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection("documents")
os.makedirs("data", exist_ok=True)

def read_pdf(path):
    """Return a list of (page_number, text) for pages that have text."""
    reader = PdfReader(path)
    pages = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
           pages.append((page_num, text))
    return pages

def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks"""
    if overlap >= size:
        raise ValueError("Overlap must be smaller than size")
    chunks = []
    start =0
    while start < len(text):
        chunks.append(text[start:start+size])
        start+= size-overlap
    return chunks

def embed(texts):
    """Turn a list of texts into embeddings usin Ollama"""
    return ollama.embed(model=EMBED_MODEL,input=texts)["embeddings"]

def ingest_pdf(path, file_name):
    """Read chunk, embed and store a pdf, REturns the number of chunks."""
    ids,docs,metas=[],[],[]
    for page_num , text in read_pdf(path):
        for i, chunk in enumerate(chunk_text(text)):
            ids.append(f"{file_name}-p{page_num}-c{i}")
            docs.append(chunk)
            metas.append({"file": file_name, "page" : page_num})

    for start in range(0,len(docs), BATCH_SIZE):
        end = start + BATCH_SIZE
        collection.upsert(
            ids=ids[start:end],
            documents=docs[start:end],
            metadatas=metas[start:end],
            embeddings=embed(docs[start:end]),
        )
    return len(docs)

def search(question, top_k=TOP_K):
    """Find the chunks closest in meaning to the question"""
    question_embedding=embed([question])[0]
    results= collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k,
    )
    hits=[]
    for text,meta,distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],

    ):
        hits.append(
            {
                "text":text,
                "file":meta["file"],
                "page":meta["page"],
                "distance":round(distance, 3),
            }
        )
    return hits

SYSTEM_PROMPT = """You answer questions using ONLY the context provided.
Rules:
- If the context does not contain the answer, reply exactly: "I don't know based on the uploaded documents."
- After each fact, cite its source like (sample.pdf, page 3).
- Do not use outside knowledge.
- Keep the answer short and clear."""


def build_context(hits):
    """Join the retrieved chunks into one text block, each labelled with its source."""
    parts = []
    for hit in hits:
        parts.append(f"[{hit['file']}, page {hit['page']}]\n{hit['text']}")
    return "\n\n---\n\n".join(parts)


def answer_question(question, model, top_k=TOP_K):
    """Retrieve relevant chunks and answer only from them."""
    if collection.count() == 0:
        return {"answer": "No documents uploaded yet.", "sources": []}

    hits = search(question, top_k)
    context = build_context(hits)

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )

    sources = [{"file": h["file"], "page": h["page"], "distance": h["distance"]} for h in hits]
    return {"answer": response["message"]["content"], "sources": sources}