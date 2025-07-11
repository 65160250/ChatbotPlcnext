import os
import logging
import requests
import time
import json
import hashlib
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- LCEL Imports ---
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
# --------------------

from langchain_ollama import OllamaLLM
from sentence_transformers import SentenceTransformer
from psycopg2 import pool
import redis

from retriever import PostgresVectorRetriever, EnhancedFlashrankRerankRetriever

# --- Configuration ---
DB_URL = os.getenv("DATABASE_URL")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Log configuration for debugging
logging.info(f"🔧 Configuration:")
logging.info(f"  DB_URL: {DB_URL}")
logging.info(f"  OLLAMA_BASE_URL: {OLLAMA_BASE_URL}")
logging.info(f"  OLLAMA_MODEL: {OLLAMA_MODEL}")
logging.info(f"  EMBED_MODEL: {EMBED_MODEL_NAME}")
logging.info(f"  REDIS_HOST: {REDIS_HOST}:{REDIS_PORT}")

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
        except requests.exceptions.RequestException as e:
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
                timeout=600  # 10 minutes timeout for model pull
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
        
        # Test pgvector extension
        cur.execute("SELECT extname FROM pg_extension WHERE extname = 'vector';")
        if not cur.fetchone():
            logging.error("❌ pgvector extension not found!")
            return False
        
        # Test documents table
        cur.execute("SELECT COUNT(*) FROM documents;")
        doc_count = cur.fetchone()[0]
        logging.info(f"✅ Database connected. Documents count: {doc_count}")
        
        cur.close()
        conn.close()
        return True
    except Exception as e:
        logging.error(f"🔥 Database connection test failed: {e}")
        return False

# --- FastAPI Lifespan: สร้าง Connection Pools และ Cache ที่นี่ ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("🚀 Starting application lifespan...")
    
    # Test database connection first
    if not test_database_connection():
        logging.error("❌ Database test failed during startup!")
        app.state.db_pool = None
    else:
        # สร้าง Connection Pool
        try:
            app.state.db_pool = pool.SimpleConnectionPool(
                1, 10, 
                dsn=DB_URL,
                # เพิ่ม connection parameters สำหรับความเสถียร
                keepalives=1,
                keepalives_idle=30,
                keepalives_interval=10,
                keepalives_count=5
            )
            logging.info("✅ Database connection pool created.")
        except Exception as e:
            logging.error(f"🔥 Failed to create database connection pool: {e}")
            app.state.db_pool = None

    # สร้าง Redis Cache Client
    try:
        app.state.redis_cache = redis.Redis(
            host=REDIS_HOST, 
            port=REDIS_PORT, 
            db=0, 
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5
        )
        app.state.redis_cache.ping()
        logging.info("✅ Redis cache connected.")
    except Exception as e:
        logging.error(f"🔥 Failed to connect to Redis: {e}")
        app.state.redis_cache = None

    # โหลด AI Models
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
    
    # Load embedding model
    try:
        app.state.embedder = SentenceTransformer(
            EMBED_MODEL_NAME, 
            cache_folder='/app/models'
        )
        logging.info(f"✅ Embedder ({EMBED_MODEL_NAME}) loaded.")
    except Exception as e:
        logging.error(f"🔥 Failed to load embedder: {e}")

    yield
    
    # ปิด Connection Pool ตอน Shutdown
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
    collection: str = "basicplcnext"

class ChatResponse(BaseModel):
    reply: str
    processing_time: float = None
    retrieval_time: float = None
    context_count: int = None

class HealthResponse(BaseModel):
    status: str
    services: dict
    timestamp: str

# --- Query Understanding ---
def preprocess_query(query: str) -> str:
    """แปลงคำย่อที่พบบ่อยในโดเมน PLCnext เพื่อเพิ่มความแม่นยำในการค้นหา"""
    abbreviations = {
        "plc": "PLCnext", 
        "hmi": "Human Machine Interface", 
        "profinet": "PROFINET",
        "i/o": "input output", 
        "gds": "Global Data Space", 
        "esm": "Execution and Synchronization Manager"
    }
    
    processed_query = query.lower()
    for abbr, full_form in abbreviations.items():
        # ใช้ word boundary เพื่อความแม่นยำ
        import re
        pattern = r'\b' + re.escape(abbr) + r'\b'
        processed_query = re.sub(pattern, full_form, processed_query)
    
    return processed_query if processed_query != query.lower() else query

# --- Enhanced Prompt ---
def build_enhanced_prompt() -> PromptTemplate:
    """Prompt ที่ปรับปรุงแล้วตามข้อเสนอแนะทั้งหมด"""
    template = """You are a specialized AI assistant for Phoenix Contact's PLCnext Technology platform.

**CONTEXT:**
{context}

**RESPONSE RULES:**
1. **GOLDEN ANSWERS PRIORITY:** If context contains "Question:...Answer:" pairs, you MUST use them verbatim.
2. **TECHNICAL PRECISION:** Include specific technical details, model numbers, and specifications.
3. **STRUCTURED ANSWERS:** For technical questions, provide a direct answer first, followed by specifications if relevant.
4. **CONTEXT ONLY:** Base answers exclusively on the provided context.
5. **NO INFO RESPONSE:** If no relevant info is found, respond with ONLY: "I could not find relevant information in the PLCnext documentation."
6. **ENGLISH ONLY:** You MUST respond ONLY in English.

**QUESTION:** {question}

**TECHNICAL ANSWER:**"""
    return PromptTemplate(input_variables=["context", "question"], template=template)

# --- Health Check Endpoint ---
@app.get("/health", response_model=HealthResponse)
async def health_check(request: Request):
    """Health check endpoint to verify all services are running."""
    services = {
        "database": False,
        "llm": False,
        "embedder": False,
        "redis": False
    }
    
    # Check database
    try:
        if request.app.state.db_pool:
            conn = request.app.state.db_pool.getconn()
            request.app.state.db_pool.putconn(conn)
            services["database"] = True
    except:
        pass
    
    # Check LLM
    services["llm"] = request.app.state.llm is not None
    
    # Check embedder
    services["embedder"] = request.app.state.embedder is not None
    
    # Check Redis
    try:
        if request.app.state.redis_cache:
            request.app.state.redis_cache.ping()
            services["redis"] = True
    except:
        pass
    
    status = "healthy" if all(services.values()) else "degraded"
    
    return HealthResponse(
        status=status,
        services=services,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
    )

# --- Main Chat Endpoint ---
@app.post("/api/chat", response_model=ChatResponse)
async def chat(fastapi_request: Request, chat_request: ChatRequest):
    """Main chat endpoint for PLCnext RAG chatbot."""
    db_pool = fastapi_request.app.state.db_pool
    llm = fastapi_request.app.state.llm
    embedder = fastapi_request.app.state.embedder
    redis_cache = fastapi_request.app.state.redis_cache

    # Check service availability
    if not all([db_pool, llm, embedder]):
        missing_services = []
        if not db_pool: missing_services.append("Database")
        if not llm: missing_services.append("LLM")
        if not embedder: missing_services.append("Embedder")
        
        error_msg = f"Required services not available: {', '.join(missing_services)}"
        logging.error(f"🔥 {error_msg}")
        raise HTTPException(status_code=503, detail=error_msg)
    
    start_time = time.perf_counter()
    
    # 1. Validate and preprocess query
    processed_msg = preprocess_query(chat_request.message.strip())
    if not processed_msg:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    
    # 2. Check cache first (if Redis is available)
    cache_key = None
    if redis_cache:
        cache_key = f"chat:{hashlib.md5(processed_msg.encode()).hexdigest()}"
        try:
            cached_response = redis_cache.get(cache_key)
            if cached_response:
                logging.info(f"📋 Cache hit for query: {processed_msg[:50]}...")
                return ChatResponse(
                    reply=cached_response,
                    processing_time=time.perf_counter() - start_time,
                    retrieval_time=0.0,
                    context_count=0
                )
        except Exception as e:
            logging.warning(f"⚠️ Cache read error: {e}")

    try:
        retrieval_start = time.perf_counter()
        
        # 3. สร้าง Retriever โดยส่ง Connection Pool เข้าไป
        base_retriever = PostgresVectorRetriever(
            connection_pool=db_pool,
            embedder=embedder,
            collection=chat_request.collection,
        )
        
        reranker_retriever = EnhancedFlashrankRerankRetriever(
            base_retriever=base_retriever
        )
        
        # 4. Build RAG chain
        prompt = build_enhanced_prompt()

        rag_chain = (
            {"context": reranker_retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        # 5. Generate response
        response_text = rag_chain.invoke(processed_msg)
        
        retrieval_time = time.perf_counter() - retrieval_start
        total_time = time.perf_counter() - start_time
        
        # 6. Cache response (if Redis is available)
        if redis_cache and cache_key:
            try:
                redis_cache.setex(cache_key, 3600, response_text)  # Cache for 1 hour
            except Exception as e:
                logging.warning(f"⚠️ Cache write error: {e}")
        
        # 7. Get context count for debugging
        try:
            context_docs = reranker_retriever.get_relevant_documents(processed_msg)
            context_count = len(context_docs)
        except:
            context_count = 0
        
        # 8. Log performance
        log_query_performance(processed_msg, response_text, retrieval_time, total_time, context_count)
        
        return ChatResponse(
            reply=response_text,
            processing_time=total_time,
            retrieval_time=retrieval_time,
            context_count=context_count
        )
        
    except Exception as e:
        logging.error(f"🔥 Chat endpoint error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"An error occurred while processing your request: {str(e)}"
        )

def log_query_performance(query: str, response: str, retrieval_time: float, total_time: float, context_count: int):
    """ฟังก์ชันสำหรับบันทึก Log การทำงาน"""
    logging.info(
        f"📊 Query Performance: "
        f"Query: '{query[:50]}...' | "
        f"Response Length: {len(response)} | "
        f"Context Count: {context_count} | "
        f"Retrieval Time: {retrieval_time:.2f}s | "
        f"Total Time: {total_time:.2f}s"
    )

# --- Additional Endpoints ---
@app.get("/api/collections")
async def get_collections(request: Request):
    """Get available document collections."""
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
    """Get database statistics."""
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

# --- Root endpoint ---
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)