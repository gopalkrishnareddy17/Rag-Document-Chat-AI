import ollama

r = ollama.chat(model="llama3.2", messages=[{"role":"user", "content":"Say hello in one line"}])
print(r["message"]["content"])