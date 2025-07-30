import os
import logging
import requests
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_ollama import OllamaLLM
from sentence_transformers import SentenceTransformer
from psycopg2 import pool

from app.retriever import PostgresVectorRetriever, EnhancedFlashrankRerankRetriever
from app.chatbot import answer_question

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
import tempfile

import pytesseract
from PIL import Image
import io

from agent_pipeline import pipeline  # import pipeline ที่เราสร้างไว้
import mimetypes

# --- Configuration ---
DB_URL = os.getenv("DATABASE_URL")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "BAAI/bge-m3")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logging.info(f"🔧 Configuration:")
logging.info(f"  DB_URL: {DB_URL}")
logging.info(f"  OLLAMA_BASE_URL: {OLLAMA_BASE_URL}")
logging.info(f"  OLLAMA_MODEL: {OLLAMA_MODEL}")
logging.info(f"  EMBED_MODEL: {EMBED_MODEL_NAME}")

# --- Helper Functions ---
def wait_for_ollama():
    """Waits for the Ollama service to be ready."""
    logging.info("🔄 Checking Ollama service readiness...")
    for attempt in range(30):
        try:
            response = requests.get(f"{OLLAMA_BASE_URL}/api/version", timeout=5)
            if response.status_code == 200:
                logging.info("✅ Ollama service is ready.")
                return True
        except requests.exceptions.RequestException:
            logging.info(f"⏳ Waiting for Ollama service... (attempt {attempt + 1}/30)")
            time.sleep(2)
    logging.error("❌ Ollama service not ready after timeout.")
    return False

def ensure_model(model_name: str):
    """Ensures the required LLM model is available in Ollama."""
    try:
        logging.info(f"🔄 Checking for LLM model: '{model_name}'")
        tags_response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
        tags_data = tags_response.json()
        available_models = [m['name'].split(":")[0] for m in tags_data.get("models", [])]
        logging.info(f"Available models: {available_models}")
        if model_name not in available_models:
            logging.warning(f"⚠️ Model '{model_name}' not found. Pulling now...")
            pull_response = requests.post(
                f"{OLLAMA_BASE_URL}/api/pull",
                json={"name": model_name},
                timeout=600
            )
            if pull_response.status_code == 200:
                logging.info(f"✅ Model '{model_name}' pulled successfully.")
            else:
                logging.error(f"❌ Failed to pull model '{model_name}': {pull_response.text}")
                return False
        else:
            logging.info(f"✅ Model '{model_name}' is available.")
        return True
    except Exception as e:
        logging.error(f"🔥 Error ensuring model '{model_name}': {e}")
        return False

def test_database_connection():
    """Test database connection and table existence."""
    try:
        import psycopg2
        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor()
        cur.execute("SELECT extname FROM pg_extension WHERE extname = 'vector';")
        if not cur.fetchone():
            logging.error("❌ pgvector extension not found!")
            return False
        cur.execute("SELECT COUNT(*) FROM documents;")
        doc_count = cur.fetchone()[0]
        logging.info(f"✅ Database connected. Documents count: {doc_count}")
        cur.close()
        conn.close()
        return True
    except Exception as e:
        logging.error(f"🔥 Database connection test failed: {e}")
        return False

# --- FastAPI Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("🚀 Starting application lifespan...")

    # Test database connection first
    if not test_database_connection():
        logging.error("❌ Database test failed during startup!")
        app.state.db_pool = None
    else:
        try:
            app.state.db_pool = pool.SimpleConnectionPool(
                1, 10,
                dsn=DB_URL,
                keepalives=1,
                keepalives_idle=30,
                keepalives_interval=10,
                keepalives_count=5
            )
            logging.info("✅ Database connection pool created.")
        except Exception as e:
            logging.error(f"🔥 Failed to create database connection pool: {e}")
            app.state.db_pool = None

    app.state.llm = None
    app.state.embedder = None

    if wait_for_ollama() and ensure_model(OLLAMA_MODEL):
        try:
            app.state.llm = OllamaLLM(
                model=OLLAMA_MODEL,
                base_url=OLLAMA_BASE_URL,
                temperature=0.0,
                timeout=60
            )
            logging.info(f"✅ LLM ({OLLAMA_MODEL}) loaded.")
        except Exception as e:
            logging.error(f"🔥 Failed to load LLM: {e}")

    try:
        app.state.embedder = SentenceTransformer(
            EMBED_MODEL_NAME,
            cache_folder='/app/models'
        )
        logging.info(f"✅ Embedder ({EMBED_MODEL_NAME}) loaded.")
    except Exception as e:
        logging.error(f"🔥 Failed to load embedder: {e}")

    yield

    if hasattr(app.state, 'db_pool') and app.state.db_pool:
        app.state.db_pool.closeall()
        logging.info("👋 Database connection pool closed.")
    logging.info("👋 Shutting down application lifespan...")

# --- App Initialization ---
app = FastAPI(
    lifespan=lifespan,
    title="PLCnext Chatbot v2.0",
    description="Advanced RAG chatbot for PLCnext Technology documentation",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# --- Pydantic Models ---
class ChatRequest(BaseModel):
    message: str
    collection: str = "plcnext"

class ChatResponse(BaseModel):
    reply: str
    processing_time: float = None
    retrieval_time: float = None
    context_count: int = None

class HealthResponse(BaseModel):
    status: str
    services: dict
    timestamp: str

# --- Health Check Endpoint ---
@app.get("/health", response_model=HealthResponse)
async def health_check(request: Request):
    services = {
        "database": False,
        "llm": False,
        "embedder": False
    }
    try:
        if request.app.state.db_pool:
            conn = request.app.state.db_pool.getconn()
            request.app.state.db_pool.putconn(conn)
            services["database"] = True
    except:
        pass
    services["llm"] = request.app.state.llm is not None
    services["embedder"] = request.app.state.embedder is not None
    status = "healthy" if all(services.values()) else "degraded"
    return HealthResponse(
        status=status,
        services=services,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
    )

# --- Main Chat Endpoint ---
@app.post("/api/chat", response_model=ChatResponse)
async def chat(fastapi_request: Request, chat_request: ChatRequest):
    db_pool = fastapi_request.app.state.db_pool
    llm = fastapi_request.app.state.llm
    embedder = fastapi_request.app.state.embedder

    result = answer_question(
        question=chat_request.message,
        db_pool=db_pool,
        llm=llm,
        embedder=embedder,
        collection=chat_request.collection,
        retriever_class=PostgresVectorRetriever,
        reranker_class=EnhancedFlashrankRerankRetriever
    )
    return ChatResponse(**result)

# --- Additional Endpoints ---
@app.get("/api/collections")
async def get_collections(request: Request):
    try:
        db_pool = request.app.state.db_pool
        if not db_pool:
            raise HTTPException(status_code=503, detail="Database not available")
        conn = db_pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT DISTINCT collection FROM documents ORDER BY collection;")
                collections = [row[0] for row in cur.fetchall()]
            return {"collections": collections}
        finally:
            db_pool.putconn(conn)
    except Exception as e:
        logging.error(f"🔥 Error fetching collections: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats")
async def get_stats(request: Request):
    try:
        db_pool = request.app.state.db_pool
        if not db_pool:
            raise HTTPException(status_code=503, detail="Database not available")
        conn = db_pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        collection,
                        COUNT(*) as doc_count,
                        AVG(LENGTH(content)) as avg_content_length
                    FROM documents 
                    GROUP BY collection
                    ORDER BY collection;
                """)
                stats = []
                for row in cur.fetchall():
                    stats.append({
                        "collection": row[0],
                        "document_count": row[1],
                        "avg_content_length": round(row[2], 2) if row[2] else 0
                    })
            return {"statistics": stats}
        finally:
            db_pool.putconn(conn)
    except Exception as e:
        logging.error(f"🔥 Error fetching stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {
        "message": "PLCnext Chatbot API v2.0",
        "endpoints": {
            "health": "/health",
            "chat": "/api/chat",
            "collections": "/api/collections",
            "stats": "/api/stats"
        }
    }

@app.post("/api/transcribe")
async def transcribe(file: UploadFile = File(...)):
    # รองรับ webm/mp3/wav
    import tempfile
    from faster_whisper import WhisperModel

    suffix = "." + file.filename.split('.')[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    model = WhisperModel("small.en", device="cpu", compute_type="float32")
    segments, _ = model.transcribe(tmp_path, language="en", beam_size=1)
    transcript = "".join([s.text for s in segments])
    return {"text": transcript}

@app.post("/api/chat-image", response_model=ChatResponse)
async def chat_image(
    request: Request,
    file: UploadFile = File(...),
    message: str = Form("")
):
    # 1. แปลงไฟล์ภาพเป็นข้อความด้วย OCR
    image = Image.open(io.BytesIO(await file.read()))
    ocr_text = pytesseract.image_to_string(image)

    # 2. รวมข้อความที่ user ถาม + ที่ได้จาก OCR
    final_question = ((message or "") + "\n" + ocr_text).strip()
    db_pool = request.app.state.db_pool
    llm = request.app.state.llm
    embedder = request.app.state.embedder
    result = answer_question(
        question=final_question,
        db_pool=db_pool,
        llm=llm,
        embedder=embedder,
        collection="plcnext",
        retriever_class=PostgresVectorRetriever,
        reranker_class=EnhancedFlashrankRerankRetriever,
    )
    return ChatResponse(**result)

@app.post("/api/agent-chat")
async def agent_chat(
    message: str = Form(""),
    file: UploadFile = File(None)
):
    # เตรียม state สำหรับ pipeline
    state = {"user_input": message}
    if file:
        content = await file.read()
        mime_type, _ = mimetypes.guess_type(file.filename)
        if mime_type and mime_type.startswith("image"):
            state["image_bytes"] = content
        elif mime_type and mime_type.startswith("audio"):
            state["audio_bytes"] = content
        else:
            return {"error": "File type not supported"}
    result = pipeline.invoke(state)
    return {"answer": result.get("llm_answer", "")}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)


