import os
import logging
import requests
import time
import math
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.eval_logging import ollama_generate_with_stats, append_eval_run
from app.ragas_eval import local_ragas_eval
from langchain_ollama import OllamaLLM
from sentence_transformers import SentenceTransformer
from psycopg2 import pool

from app.retriever import PostgresVectorRetriever, EnhancedFlashrankRerankRetriever
from app.chatbot import answer_question
from app.ragas_eval import local_ragas_eval


from agent_pipeline import pipeline
import mimetypes
import importlib
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ---- Config ----
DB_URL = os.getenv("DATABASE_URL")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "BAAI/bge-m3")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logging.info("🔧 Configuration:")
logging.info(f"  DB_URL: {DB_URL}")
logging.info(f"  OLLAMA_BASE_URL: {OLLAMA_BASE_URL}")
logging.info(f"  OLLAMA_MODEL: {OLLAMA_MODEL}")
logging.info(f"  EMBED_MODEL: {EMBED_MODEL_NAME}")

# -------------------------
# Helpers
# -------------------------
def wait_for_ollama():
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
    try:
        logging.info(f"🔄 Checking for LLM model: '{model_name}'")
        tags_response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
        tags_data = tags_response.json()
        available = [m['name'].split(":")[0] for m in tags_data.get("models", [])]
        logging.info(f"Available models: {available}")
        if model_name not in available:
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

# -------------------------
# JSON sanitizer: NaN/Inf -> None
# -------------------------
def _is_bad_float(x: Any) -> bool:
    return isinstance(x, float) and (math.isnan(x) or math.isinf(x))

def sanitize_json(obj: Any):
    if _is_bad_float(obj):
        return None
    if isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_json(v) for v in obj]
    return obj

# ---- RAGAS helpers ----
def _to_1_10(x):
    # handle None/NaN/Inf robustly
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return None
    try:
        x = max(0.0, min(1.0, float(x))) * 10.0
        return round(max(1.0, x), 2)
    except Exception:
        return None

def _load_ragas_adapters():
    trials = [
        ("ragas.integrations.langchain", "LangchainLLM", "LangchainEmbeddings"),
        ("ragas.integrations.langchain", "LangChainLLM", "LangChainEmbeddings"),
        ("ragas.llms", "LangchainLLM", None),
        ("ragas.llms", "LangChainLLM", None),
    ]
    for mod, llm_name, emb_name in trials:
        try:
            m = importlib.import_module(mod)
            LLMClass = getattr(m, llm_name)
            if emb_name:
                EmbClass = getattr(m, emb_name)
            else:
                me = importlib.import_module("ragas.embeddings")
                EmbClass = getattr(me, "LangchainEmbeddings", None) or getattr(me, "LangChainEmbeddings")
            return LLMClass, EmbClass
        except Exception:
            continue
    return None, None

def compute_ragas_single(question, contexts, answer):
    """
    ใช้ RAGAS แบบยืดหยุ่น:
    - โหลด adapter ได้หลาย path/ชื่อคลาส (รองรับต่างเวอร์ชัน)
    - บังคับ LLM = Ollama (ไม่แตะ OpenAI)
    - ถ้า LLM path ล้ม → fallback no‑LLM (context_precision/recall)
    """
    import math, importlib, os
    try:
        from datasets import Dataset
        from ragas import evaluate
        # นำ metric มาแบบฟังก์ชัน (ไม่ bind อะไรใน constructor)
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from langchain_ollama import OllamaLLM
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except Exception as e:
        return {"error": "ragas_unavailable", "detail": f"imports_failed: {e}"}

    ds = Dataset.from_list([{
        "question": question or "",
        "contexts": contexts or [],
        "answer": answer or "",
        "ground_truth": ""
    }])

    # ----- ตัวโหลด adapter แบบยืดหยุ่น (รองรับหลายเวอร์ชัน/หลายชื่อคลาส)
    def _load_ragas_adapters():
        trials = [
            ("ragas.integrations.langchain", "LangchainLLM", "LangchainEmbeddings"),
            ("ragas.integrations.langchain", "LangChainLLM", "LangChainEmbeddings"),
            ("ragas.llms", "LangchainLLM", None),
            ("ragas.llms", "LangChainLLM", None),
        ]
        for mod, llm_name, emb_name in trials:
            try:
                m = importlib.import_module(mod)
                LLMClass = getattr(m, llm_name)
                if emb_name:
                    EmbClass = getattr(m, emb_name)
                else:
                    me = importlib.import_module("ragas.embeddings")
                    EmbClass = getattr(me, "LangchainEmbeddings", None) or getattr(me, "LangChainEmbeddings")
                return LLMClass, EmbClass
            except Exception:
                continue
        return None, None

    LLMAdapter, EmbAdapter = _load_ragas_adapters()
    use_llm = LLMAdapter is not None and EmbAdapter is not None

    try:
        if use_llm:
            judge = LLMAdapter(
                OllamaLLM(
                    model=os.getenv("RAGAS_JUDGE_MODEL", os.getenv("OLLAMA_MODEL", "llama3.2")),
                    base_url=os.getenv("OLLAMA_BASE_URL", "http://ollama:11434"),
                    temperature=0
                )
            )
            emb = EmbAdapter(
                HuggingFaceEmbeddings(
                    model_name=os.getenv("EVAL_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
                    encode_kwargs={"normalize_embeddings": True}
                )
            )
            res = evaluate(ds, metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
                           llm=judge, embeddings=emb)
        else:
            # no‑LLM: คิดเฉพาะ context_* (กันฉุกเฉิน ไม่ต้องใช้ LLM)
            hf = HuggingFaceEmbeddings(
                model_name=os.getenv("EVAL_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
                encode_kwargs={"normalize_embeddings": True}
            )
            class _EmbWrap:
                def __init__(self, inner): self.inner = inner
                def embed_documents(self, texts): return self.inner.embed_documents(texts)
                def embed_query(self, text): return self.inner.embed_query(text)
            res = evaluate(ds, metrics=[context_precision, context_recall], embeddings=_EmbWrap(hf))
    except Exception as e:
        # ถ้า LLM path ล้มกลางคัน → fallback no‑LLM
        try:
            from ragas.metrics import context_precision, context_recall
            hf = HuggingFaceEmbeddings(
                model_name=os.getenv("EVAL_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
                encode_kwargs={"normalize_embeddings": True}
            )
            class _EmbWrap:
                def __init__(self, inner): self.inner = inner
                def embed_documents(self, texts): return self.inner.embed_documents(texts)
                def embed_query(self, text): return self.inner.embed_query(text)
            res = evaluate(ds, metrics=[context_precision, context_recall], embeddings=_EmbWrap(hf))
        except Exception as ee:
            return {"error": "ragas_eval_failed", "detail": str(ee)}

    # ---- ดึงผล + scale 1-10 อย่างทนทาน ----
    try:
        results_obj = getattr(res, "results", None)
        if results_obj is not None:
            try:
                row = results_obj[0]
            except Exception:
                row = results_obj.to_pandas().iloc[0].to_dict()
        else:
            row = {
                "answer_relevancy": res.get("answer_relevancy"),
                "faithfulness": res.get("faithfulness"),
                "context_precision": res.get("context_precision"),
                "context_recall": res.get("context_recall"),
            }
    except Exception:
        row = {"answer_relevancy": None, "faithfulness": None, "context_precision": None, "context_recall": None}

    def _to_1_10(x):
        if x is None: return None
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)): return None
        x = max(0.0, min(1.0, float(x))) * 10.0
        return round(max(1.0, x), 2)

    raw = {
        "answer_relevancy": row.get("answer_relevancy"),
        "faithfulness": row.get("faithfulness"),
        "context_precision": row.get("context_precision"),
        "context_recall": row.get("context_recall"),
    }
    scaled = {
        "relevance": _to_1_10(raw["answer_relevancy"]),
        "faithfulness": _to_1_10(raw["faithfulness"]),
        "context_precision": _to_1_10(raw["context_precision"]),
        "context_recall": _to_1_10(raw["context_recall"]),
    }
    return {"raw": raw, "scale_1_10": scaled}



# ---- FastAPI Lifespan ----
@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("🚀 Starting application lifespan...")

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

# ---- App init ----
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

# ---- Models ----
class ChatRequest(BaseModel):
    message: str
    collection: str = "plcnext"

class ChatResponse(BaseModel):
    reply: str
    processing_time: float | None = None
    retrieval_time: float | None = None
    context_count: int | None = None
    ragas: dict | None = None

class HealthResponse(BaseModel):
    status: str
    services: dict
    timestamp: str

# ---- Endpoints ----
@app.get("/health", response_model=HealthResponse)
async def health_check(request: Request):
    services = {"database": False, "llm": False, "embedder": False}
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
    return HealthResponse(status=status, services=services, timestamp=time.strftime("%Y-%m-%d %H:%M:%S"))

@app.post("/api/chat", response_model=ChatResponse)
async def chat(fastapi_request: Request, chat_request: ChatRequest):
    db_pool = fastapi_request.app.state.db_pool
    llm = fastapi_request.app.state.llm
    embedder = fastapi_request.app.state.embedder

    # 1) ตอบคำถาม
    result = answer_question(
        question=chat_request.message,
        db_pool=db_pool,
        llm=llm,
        embedder=embedder,
        collection=chat_request.collection,
        retriever_class=PostgresVectorRetriever,
        reranker_class=EnhancedFlashrankRerankRetriever
    )

    # 2) ประเมินด้วย RAGAS แบบโลคัล (ถ้าเปิดใช้)
    ragas_payload = None
    try:
        if os.getenv("EVAL_WITH_RAGAS", "false").lower() in ("1", "true", "yes"):
            # ดึง contexts อีกรอบ แบบ list[str]
            base_ret = PostgresVectorRetriever(connection_pool=fastapi_request.app.state.db_pool, embedder=embedder, collection=chat_request.collection)
            reranker = EnhancedFlashrankRerankRetriever(base_retriever=base_ret)
            docs = reranker.get_relevant_documents(chat_request.message)
            contexts = [d.page_content for d in docs]

            ragas_payload = local_ragas_eval(
                question=chat_request.message,
                answer=result.get("reply", ""),
                contexts=contexts
            )
    except Exception as e:
        ragas_payload = {"status": "skipped", "reason": str(e)}

    return ChatResponse(
        reply=result["reply"],
        processing_time=result.get("processing_time"),
        retrieval_time=result.get("retrieval_time"),
        context_count=result.get("context_count"),
        ragas=ragas_payload
    )

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
    import io
    import pytesseract
    from PIL import Image

    image = Image.open(io.BytesIO(await file.read()))
    ocr_text = pytesseract.image_to_string(image)

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
    file: UploadFile = File(None),
    log_eval: bool = Form(False),
    return_ragas_metrics: bool = Form(False)
):
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

    start_time = time.perf_counter()
    result = pipeline.invoke(state)
    total_time = time.perf_counter() - start_time

    contexts = result.get("contexts_list") or []
    eval_info = None

    if log_eval:
        prompt = (
            "จงตอบคำถามต่อไปนี้โดยอ้างอิงเฉพาะข้อมูลจากบริบทที่ให้เท่านั้น ห้ามเดา\n\n"
            "บริบท:\n" + "\n".join(f"- {c}" for c in contexts) +
            f"\n\nคำถาม: {message}\nคำตอบ:"
        )
        eval_answer, timing = ollama_generate_with_stats(
            prompt, model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL
        )
        append_eval_run(
            os.getenv("EVAL_LOG_FILE", "eval_runs.jsonl"),
            {
                "question": message,
                "contexts": contexts,
                "answer": eval_answer,
                "ground_truth": "",
                "timing": timing,
            }
        )
        eval_info = {
            "logged": True,
            "timing": timing,
            "contexts_count": len(contexts),
            "eval_answer_preview": (eval_answer[:200] + "...") if eval_answer else ""
        }

    ragas_metrics = None
    if return_ragas_metrics:
        try:
            ragas_metrics = local_ragas_eval(
                question=message,
                contexts=contexts,
                answer=result.get("llm_answer", "")
            )
        except Exception as e:
            ragas_metrics = {"status": "skipped", "reason": str(e)}

    response = {
        "reply": result.get("llm_answer", ""),
        "processing_time": result.get("processing_time", total_time),
        "retrieval_time": result.get("retrieval_time", None),
        "context_count": result.get("context_count", None),
        "contexts": contexts,
        "eval": eval_info,
        "ragas": ragas_metrics
    }
    return sanitize_json(response)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
