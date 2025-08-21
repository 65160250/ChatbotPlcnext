import os, asyncio, json, math, importlib, sys, traceback
os.environ.setdefault("ANYIO_BACKEND", "asyncio")
try:
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
except Exception:
    pass

import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from langchain_ollama import OllamaLLM
from langchain_community.embeddings import HuggingFaceEmbeddings

EVAL_FILE = os.getenv("EVAL_FILE", "eval_runs.jsonl")
OUT_CSV   = os.getenv("RAGAS_OUT_CSV", "ragas_report.csv")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
RAGAS_JUDGE_MODEL = os.getenv("RAGAS_JUDGE_MODEL", os.getenv("OLLAMA_MODEL", "llama3.2"))
EVAL_EMBED_MODEL = os.getenv("EVAL_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
FORCE_NO_LLM = os.getenv("RAGAS_NO_LLM", "0") == "1"

def log(*a): print("[run_ragas]", *a, file=sys.stderr)

def load_jsonl(path):
    items = []
    if not os.path.exists(path):
        log(f"ไม่พบไฟล์ {path}")
        return items
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items

def to_1_10(x):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return None
    x = max(0.0, min(1.0, float(x))) * 10.0
    return round(max(1.0, x), 2)

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
            log(f"ใช้ adapter จาก {mod}.{llm_name}")
            return LLMClass, EmbClass
        except Exception as e:
            continue
    log("ไม่พบ LangChain adapter ที่ใช้ได้ใน ragas")
    return None, None

def evaluate_with_llm(ds):
    # Judge = Ollama (ไม่ยุ่ง OpenAI)
    judge = None; emb = None
    try:
        LLMAdapter, EmbAdapter = _load_ragas_adapters()
        if not (LLMAdapter and EmbAdapter):
            raise RuntimeError("no_adapter")
        judge = LLMAdapter(
            OllamaLLM(
                model=RAGAS_JUDGE_MODEL,
                base_url=OLLAMA_BASE_URL,
                temperature=0
            )
        )
        emb = EmbAdapter(
            HuggingFaceEmbeddings(
                model_name=EVAL_EMBED_MODEL,
                encode_kwargs={"normalize_embeddings": True}
            )
        )
        log(f"เริ่ม evaluate โหมด LLM judge={RAGAS_JUDGE_MODEL} via Ollama")
        return evaluate(ds, metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
                        llm=judge, embeddings=emb)
    except Exception as e:
        # ถ้าโดน error api_key / import อะไร ให้ fallback
        log("LLM evaluate ล้มเหลว → fallback เป็น no‑LLM")
        traceback.print_exc()
        return None

def evaluate_no_llm(ds):
    log("เริ่ม evaluate โหมด no‑LLM (เฉพาะ context_* metrics)")
    hf = HuggingFaceEmbeddings(
        model_name=EVAL_EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True}
    )
    class _EmbWrap:
        def __init__(self, inner): self.inner = inner
        def embed_documents(self, texts): return self.inner.embed_documents(texts)
        def embed_query(self, text): return self.inner.embed_query(text)
    emb = _EmbWrap(hf)
    return evaluate(ds, metrics=[context_precision, context_recall], embeddings=emb)

def main():
    log(f"RAGAS_NO_LLM={FORCE_NO_LLM} | RAGAS_JUDGE_MODEL={RAGAS_JUDGE_MODEL} | OLLAMA_BASE_URL={OLLAMA_BASE_URL}")
    items = load_jsonl(EVAL_FILE)
    if not items:
        print("ยังไม่มีรายการประเมินใน eval_runs.jsonl")
        return

    ds = Dataset.from_list([{
        "question": it.get("question",""),
        "contexts": it.get("contexts",[]),
        "answer":   it.get("answer",""),
        "ground_truth": it.get("ground_truth","") or "",
    } for it in items])

    res = None
    if not FORCE_NO_LLM:
        res = evaluate_with_llm(ds)
    if res is None:  # ใช้ no‑LLM เมื่อสั่งบังคับ หรือเมื่อ LLM mode fail
        res = evaluate_no_llm(ds)

    results = getattr(res, "results", None)
    if results is not None and hasattr(results, "to_pandas"):
        pdf = results.to_pandas()
        rename = {
            "answer_relevancy": "relevance_1_10",
            "faithfulness": "faithfulness_1_10",
            "context_recall": "context_recall_1_10",
            "context_precision": "context_precision_1_10",
        }
        for src, dst in rename.items():
            if src in pdf.columns:
                pdf[dst] = pdf[src].map(to_1_10)
        keep = ["question","answer"] + [v for v in rename.values() if v in pdf.columns]
        pdf[keep].to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    else:
        # บางเวอร์ชัน ragas คืนค่า aggregate dict
        avg = {k: res.get(k) for k in ["answer_relevancy","faithfulness","context_recall","context_precision"] if hasattr(res, "get")}
        out = {
            "question": "(average only)",
            "answer": "",
            "relevance_1_10": to_1_10(avg.get("answer_relevancy")) if avg else None,
            "faithfulness_1_10": to_1_10(avg.get("faithfulness")) if avg else None,
            "context_recall_1_10": to_1_10(avg.get("context_recall")) if avg else None,
            "context_precision_1_10": to_1_10(avg.get("context_precision")) if avg else None,
        }
        pd.DataFrame([out]).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print(f"บันทึก {OUT_CSV} แล้ว (judge={RAGAS_JUDGE_MODEL}, no_llm={FORCE_NO_LLM})")

if __name__ == "__main__":
    main()
