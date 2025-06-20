# backend/app.py
from flask import Flask, request, jsonify
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from sentence_transformers import SentenceTransformer
from retriever import PostgresVectorRetriever
import os

app = Flask(__name__)

DB_URL = os.getenv("DATABASE_URL")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")

llm = OllamaLLM(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.8)
embedder = SentenceTransformer("intfloat/multilingual-e5-large")

retriever = PostgresVectorRetriever(
    connection_string=DB_URL,
    table="documents",
    embedder=embedder,
    collection="plcnext"
)

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
คุณคือผู้ช่วยด้าน PLCnext

บริบท:
{context}

คำถาม:
{question}
If the question is in Thai, respond in Thai.
If the question is in English, respond in English.

ตอบโดยอิงจากบริบทที่ให้เท่านั้น หากไม่มีข้อมูล ให้ตอบว่าไม่พบข้อมูล

Answer based on the provided context only. If no information is available, respond with: **"No information found."**
"""
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt}
)

@app.route("/api/chat", methods=["POST"])
def chat():
    msg = request.json.get("message", "").strip()
    if not msg:
        return jsonify({"error": "empty"}), 400
    try:
        result = qa.run(msg)
        return jsonify({"reply": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/ping")
def ping():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
