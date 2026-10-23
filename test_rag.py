from rag import read_pdf, chunk_text,embed,ingest_pdf, collection, search, answer_question

# pages = read_pdf("data/sample.pdf")
# print("pages with text : ", len(pages))

# first_page_num, first_text = pages[0]
# print("Page", first_page_num, "Preview:")
# print(first_text[:300])

# pages = read_pdf("data/sample.pdf")
# for i in pages:
#     page_num, text = i

#     chunks = chunk_text(text)
#     print("Page", page_num, "length:", len(text), "characters")
#     print("Number of chunks : ", len(chunks))

#     for i, chunk in enumerate(chunks):
#         print(f"\n--- Chunk {i} ({len(chunk)} chars0---)")
#         print(chunk[:150])

# texts = ["The acr is expensive", "The vehicle closts a lot", "I like mangoes"]
# vectors = embed(texts)

# print("Numnber of embeddings ", len(vectors))
# for i in vectors:
#     print("Numbers per embedding: ",len(i))

# print("First 5 numbers : ", vectors[0][:5])

# def similarity(a, b):
#     dot = sum(x*y for x,y in zip(a,b))
#     norm_a = sum(x*x for x in a)**0.5
#     nomr_b = sum(y*y for y in b)**0.5
#     return dot/(norm_a * nomr_b)

# print("Car vs Vehicle : " , round(similarity(vectors[0], vectors[1]), 3))
# print("Car vs MAango : " , round(similarity(vectors[0], vectors[2]), 3))

# count = ingest_pdf("data/sample.pdf", "sample.pdf")
# print("chunks stored : ", count)
# print("Total in database : " , collection.count())

# sample = collection.get(limit=3)
# for doc_id, meta in zip(sample["ids"], sample["metadatas"]):
#     print(doc_id,meta)

# question = "Candidate name"
# hits = search(question)

# for i, hit in enumerate(hits):
#     print(f"\n#{i} {hit["file"]} distance {hit["distance"]}")
#     print(hit["text"][:200])

questions = [
    "Candidate Details",     # change to a real question about your PDF
    "Technology is in document?", # unrelated question
]

for q in questions:
    result = answer_question(q, model="llama3.2")
    print("\nQ:", q)
    print("A:", result["answer"])
    print("Sources:", [(s["file"], s["page"]) for s in result["sources"]])