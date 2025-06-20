# backend/embed.py
import os, argparse, requests, time, hashlib
from dotenv import load_dotenv
# from langchain_community.embeddings import OllamaEmbeddings
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, TextLoader, JSONLoader
import psycopg2

load_dotenv()
DB_URL = os.getenv("DATABASE_URL")
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

def wait_for_ollama():  # ตรวจสอบให้แน่ใจว่า ollama พร้อม
    for _ in range(20):
        try:
            if requests.get(f"{OLLAMA_URL}/api/version", timeout=5).status_code == 200:
                return True
        except: time.sleep(2)
    return False

def ensure_model(model):
    try:
        tags = requests.get(f"{OLLAMA_URL}/api/tags").json()
        if model not in [m['name'].split(":")[0] for m in tags.get("models", [])]:
            r = requests.post(f"{OLLAMA_URL}/api/pull", json={"name": model}, timeout=300)
            return r.status_code == 200
        return True
    except: return False

def load_docs(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf": return PyPDFLoader(path).load()
    if ext == ".csv": return CSVLoader(path).load()
    if ext == ".txt": return TextLoader(path, encoding="utf-8").load()
    if ext == ".json": return JSONLoader(path, jq_schema=".content").load()
    raise ValueError("Unsupported file type")

def insert_to_postgres(docs, embedder, collection):
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    
    for doc in docs:
        content = doc.page_content.strip()
        vector = embedder.encode(content).tolist()  # << ใช้ sentence-transformer
        hash_ = hashlib.sha256(content.encode()).hexdigest()
        try:
            cur.execute("""
                INSERT INTO documents (content, embedding, collection, hash)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (hash) DO NOTHING;
            """, (content, vector, collection, hash_))
        except Exception as e:
            print("❌ Insert error:", e)
    conn.commit()
    conn.close()

def embed_file(file_path, collection="default"):
    docs = load_docs(file_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    embedder = SentenceTransformer("intfloat/multilingual-e5-large")  # ✅ ใช้ตัวเดียวกับ retriever
    insert_to_postgres(chunks, embedder, collection)
    print(f"✅ {len(chunks)} chunks embedded to collection: {collection}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Path to file")
    parser.add_argument("--collection", default="default")
    args = parser.parse_args()

    if not wait_for_ollama(): exit("❌ Ollama not ready")
    if not ensure_model(OLLAMA_MODEL): exit("❌ Model not found")
    if not os.path.exists(args.file): exit("❌ File not found")

    embed_file(args.file, args.collection)
