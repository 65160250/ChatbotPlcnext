from langgraph.graph import StateGraph, END
from langchain_ollama import OllamaLLM
from app.retriever import PostgresVectorRetriever, EnhancedFlashrankRerankRetriever
from app.chatbot import build_enhanced_prompt
from sentence_transformers import SentenceTransformer
import os
from psycopg2 import pool
from typing import TypedDict, Optional

from PIL import Image
import pytesseract
import io

from faster_whisper import WhisperModel
import tempfile

# --- config ---
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

# --- State: เพิ่ม audio_bytes ด้วย ---
class AgentState(TypedDict):
    user_input: str
    input_type: Optional[str]
    context: Optional[str]
    llm_answer: Optional[str]
    image_bytes: Optional[bytes]
    audio_bytes: Optional[bytes]   # << เพิ่มตรงนี้

# --- Node: classifier ---
def classifier_node(state, config=None):
    if state.get("audio_bytes"):
        return {"input_type": "audio"}
    elif state.get("image_bytes"):
        return {"input_type": "image"}
    else:
        return {"input_type": "text"}

# --- Node: OCR (image) ---
def ocr_node(state, config=None):
    image_bytes = state["image_bytes"]
    image = Image.open(io.BytesIO(image_bytes))
    ocr_text = pytesseract.image_to_string(image)
    return {"user_input": ocr_text}

# --- Node: Audio to Text (audio) ---
def audio2text_node(state, config=None):
    audio_bytes = state["audio_bytes"]
    # save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpf:
        tmpf.write(audio_bytes)
        tmp_path = tmpf.name
    model = WhisperModel("small.en", device="cpu", compute_type="float32")
    segments, _ = model.transcribe(tmp_path, language="en", beam_size=1)
    transcript = "".join([s.text for s in segments])
    return {"user_input": transcript}

# --- Node: Retrieval ---
def retrieval_node(state, config=None):
    question = state["user_input"]
    base_retriever = PostgresVectorRetriever(
        connection_pool=db_pool,
        embedder=embedder,
        collection="plcnext",
    )
    reranker = EnhancedFlashrankRerankRetriever(base_retriever=base_retriever)
    docs = reranker.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])
    return {"context": context}

# --- Node: LLM ---
def llm_node(state, config=None):
    question = state["user_input"]
    context = state["context"]
    prompt = build_enhanced_prompt()
    message = prompt.format(context=context, question=question)
    response = llm.invoke(message)
    return {"llm_answer": response}

# --- Graph ---
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
# หลัง ocr กับ audio2text ให้ไป retrieval ต่อ
graph.add_edge("ocr", "retrieval")
graph.add_edge("audio2text", "retrieval")
graph.add_edge("retrieval", "llm")
graph.add_edge("llm", END)

pipeline = graph.compile()

# --- ทดสอบรัน (เลือก mode ได้) ---
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
    print("AI:", result["llm_answer"])
