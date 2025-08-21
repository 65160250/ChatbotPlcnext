from langgraph.graph import StateGraph, END
from langchain_ollama import OllamaLLM
from app.retriever import PostgresVectorRetriever, EnhancedFlashrankRerankRetriever
from app.chatbot import build_enhanced_prompt
from sentence_transformers import SentenceTransformer
import os
from psycopg2 import pool
from typing import TypedDict, Optional
import time

from PIL import Image
import pytesseract
import io

from faster_whisper import WhisperModel
import tempfile

# === CONFIG ===
DB_URL = os.getenv("DATABASE_URL")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "BAAI/bge-m3")

llm = OllamaLLM(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0.0,
    timeout=60
)
embedder = SentenceTransformer(EMBED_MODEL_NAME, cache_folder='/app/models')
db_pool = pool.SimpleConnectionPool(
    1, 10,
    dsn=DB_URL,
    keepalives=1,
    keepalives_idle=30,
    keepalives_interval=10,
    keepalives_count=5
)

# === STATE SCHEMA ===
class AgentState(TypedDict):
    user_input: str
    input_type: Optional[str]
    context: Optional[str]
    llm_answer: Optional[str]
    image_bytes: Optional[bytes]
    audio_bytes: Optional[bytes]
    retrieval_time: Optional[float]
    context_count: Optional[int]
    processing_time: Optional[float]
    contexts_list: Optional[list[str]]

# === NODES ===

def classifier_node(state, config=None):
    """ตรวจสอบประเภท input เพื่อ branch ไป node ที่ถูกต้อง"""
    if state.get("audio_bytes"):
        return {"input_type": "audio"}
    elif state.get("image_bytes"):
        return {"input_type": "image"}
    else:
        return {"input_type": "text"}

def ocr_node(state, config=None):
    """แปลง image เป็น text ด้วย OCR"""
    image_bytes = state["image_bytes"]
    image = Image.open(io.BytesIO(image_bytes))
    ocr_text = pytesseract.image_to_string(image)
    return {"user_input": ocr_text}

def audio2text_node(state, config=None):
    """แปลง audio เป็นข้อความ (ASR)"""
    audio_bytes = state["audio_bytes"]
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpf:
        tmpf.write(audio_bytes)
        tmp_path = tmpf.name
    model = WhisperModel("small.en", device="cpu", compute_type="float32")
    segments, _ = model.transcribe(tmp_path, language="en", beam_size=1)
    transcript = "".join([s.text for s in segments])
    return {"user_input": transcript}

def retrieval_node(state, config=None):
    question = state["user_input"]
    start = time.perf_counter()
    base_retriever = PostgresVectorRetriever(
        connection_pool=db_pool,
        embedder=embedder,
        collection="plcnext",
    )
    reranker = EnhancedFlashrankRerankRetriever(base_retriever=base_retriever)
    docs = reranker.invoke(question)

    contexts_list = [d.page_content for d in docs]     # << เพิ่ม
    context = "\n\n".join(contexts_list)
    retrieval_time = time.perf_counter() - start
    context_count = len(docs)

    return {
        "context": context,
        "contexts_list": contexts_list,                # << เพิ่ม
        "retrieval_time": retrieval_time,
        "context_count": context_count,
        "user_input": question
    }



def llm_node(state, config=None):
    """ตอบคำถามด้วย LLM พร้อมจับเวลา"""
    question = state["user_input"]
    context = state["context"]
    prompt = build_enhanced_prompt()
    start = time.perf_counter()
    message = prompt.format(context=context, question=question)
    response = llm.invoke(message)
    total_time = time.perf_counter() - start
    return {
        "llm_answer": response,
        "processing_time": total_time
    }

# === GRAPH COMPOSITION ===
graph = StateGraph(AgentState)

graph.add_node("classifier", classifier_node)
graph.add_node("ocr", ocr_node)
graph.add_node("audio2text", audio2text_node)
graph.add_node("retrieval", retrieval_node)
graph.add_node("llm", llm_node)

graph.set_entry_point("classifier")
graph.add_conditional_edges(
    "classifier",
    lambda state: state["input_type"],
    {
        "image": "ocr",
        "audio": "audio2text",
        "text": "retrieval"
    }
)
graph.add_edge("ocr", "retrieval")
graph.add_edge("audio2text", "retrieval")
graph.add_edge("retrieval", "llm")
graph.add_edge("llm", END)

pipeline = graph.compile()

# === FOR TEST VIA COMMAND LINE ===
if __name__ == "__main__":
    mode = input("Input type (text/image/audio): ").strip()
    state = {}
    if mode == "text":
        state["user_input"] = input("User: ")
    elif mode == "image":
        with open("path/to/image.png", "rb") as f:
            state["user_input"] = ""
            state["image_bytes"] = f.read()
    elif mode == "audio":
        with open("path/to/audio.wav", "rb") as f:
            state["user_input"] = ""
            state["audio_bytes"] = f.read()
    else:
        print("Not supported.")
        exit()
    result = pipeline.invoke(state)
    print("AI:", result.get("llm_answer"))
    print("Processing time:", result.get("processing_time"))
    print("Retrieval time:", result.get("retrieval_time"))
    print("Context count:", result.get("context_count"))
