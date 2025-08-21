# app/ragas_eval.py
import os, re
from typing import List, Dict, Any

def _to_float(x):
    try:
        return float(getattr(x, "value", x))
    except Exception:
        return None

def _extract_scores_dataframe_like(result_obj) -> Dict[str, Any]:
    """
    ดึงคะแนนจากผลลัพธ์ ragas โดยพยายามอ่านผ่าน .to_pandas() ก่อน (ทนทุกเวอร์ชัน)
    แล้วค่อย fallback เป็น .results หรือ mapping
    """
    metric_keys = ["context_precision", "faithfulness", "answer_relevancy", "context_recall"]

    # 1) to_pandas()
    try:
        if hasattr(result_obj, "to_pandas"):
            df = result_obj.to_pandas()
            if len(df) > 0:
                row = df.iloc[0].to_dict()
                return {k: _to_float(row.get(k)) for k in metric_keys}
    except Exception:
        pass

    # 2) results
    try:
        if hasattr(result_obj, "results"):
            rows = result_obj.results
            if isinstance(rows, list) and rows:
                row = rows[0]
                getv = row.get if hasattr(row, "get") else (lambda kk: getattr(row, kk, None))
                return {k: _to_float(getv(k)) for k in metric_keys}
    except Exception:
        pass

    # 3) mapping
    try:
        as_dict = dict(result_obj)
        return {k: _to_float(as_dict.get(k)) for k in metric_keys}
    except Exception:
        pass

    # 4) attributes
    return {k: _to_float(getattr(result_obj, k, None)) for k in metric_keys}

def _guess_ground_truth(question: str, contexts: List[str]) -> str | None:
    """
    พยายามเด้ง ground_truth จากบริบทที่เป็นรูป Q/A:
    - หา block "Question: ...\nAnswer: ..." ที่ 'คล้าย' กับคำถาม
    - ถ้าพบ คืนข้อความหลัง "Answer:" เป็น ground_truth
    """
    q = (question or "").strip().lower()
    qa_pat = re.compile(r"Question:\s*(?P<q>.+?)\s*[\r\n]+Answer:\s*(?P<a>.+)", re.IGNORECASE | re.DOTALL)
    best = None
    for c in contexts or []:
        m = qa_pat.search(c or "")
        if not m:
            continue
        qx = (m.group("q") or "").strip().lower()
        ax = (m.group("a") or "").strip()
        # heuristic: ถ้าคำถามซ้ำกันหรือตัดคำแล้วมีส่วนร่วมเยอะ ให้ถือว่า match
        overlap = sum(t in qx for t in q.split() if len(t) > 2)
        if q == qx or overlap >= max(2, len(q.split()) // 3):
            best = ax
            break
    return best

def local_ragas_eval(question: str, answer: str, contexts: List[str]) -> Dict[str, Any]:
    """
    Evaluate RAG outputs using RAGAS (อัตโนมัติเลือก judge):
    - มี OPENAI_API_KEY → ใช้ OpenAI (แม่นขึ้น)
    - ไม่มี → ใช้ Ollama+OllamaEmbeddings
    - ถ้ามี ground_truth (เด้งจาก contexts) จะคำนวณครบ 4 metric
      ถ้าไม่มี จะคำนวณเฉพาะ faithfulness / answer_relevancy
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            context_precision, faithfulness, answer_relevancy, context_recall
        )

        use_openai = bool(os.getenv("OPENAI_API_KEY"))

        if use_openai:
            from langchain_openai import ChatOpenAI, OpenAIEmbeddings
            llm = ChatOpenAI(model=os.getenv("RAGAS_JUDGE_MODEL", "gpt-4o-mini"), temperature=0)
            embeddings = OpenAIEmbeddings(model=os.getenv("EVAL_EMBED_MODEL", "text-embedding-3-large"))
            judge_name = "openai"
        else:
            from langchain_community.chat_models import ChatOllama
            from langchain_community.embeddings import OllamaEmbeddings
            llm = ChatOllama(
                model=os.getenv("OLLAMA_MODEL", "llama3.2"),
                base_url=os.getenv("OLLAMA_BASE_URL", "http://ollama:11434"),
            )
            embeddings = OllamaEmbeddings(
                model=os.getenv("OLLAMA_EMBED_MODEL", os.getenv("OLLAMA_MODEL", "llama3.2")),
                base_url=os.getenv("OLLAMA_BASE_URL", "http://ollama:11434"),
            )
            judge_name = "ollama"

        gt = _guess_ground_truth(question, contexts)  # เด้ง ground_truth อัตโนมัติถ้าเจอ
        has_gt = bool(gt and gt.strip())

        ds = Dataset.from_dict({
            "question": [question or ""],
            "contexts": [contexts or []],
            "answer":   [answer or ""],
            # ถ้ามี gt ใส่จริง, ถ้าไม่มี ใส่ "" ก็ได้ (แต่เราจะเลือก metric ให้เหมาะ)
            "ground_truth": [gt or ""]
        })

        # เลือก metric ให้เหมาะกับข้อมูลที่มี
        metrics = [faithfulness, answer_relevancy]
        if has_gt:
            metrics += [context_precision, context_recall]

        result = evaluate(ds, metrics=metrics, llm=llm, embeddings=embeddings)
        scores = _extract_scores_dataframe_like(result)

        # ให้ key ครบทั้ง 4 เสมอ แต่ถ้าไม่ได้คำนวณก็เป็น None
        for k in ["context_precision", "faithfulness", "answer_relevancy", "context_recall"]:
            if k not in scores or scores[k] is None:
                if (k in ["context_precision", "context_recall"]) and not has_gt:
                    scores[k] = None  # ไม่คำนวณ เพราะไม่มี ground_truth
                else:
                    # กันเวอร์ชันแปลก ๆ
                    scores[k] = None

        return {"status": "ok", "scores": scores, "judge": judge_name, "has_ground_truth": has_gt}
    except Exception as e:
        return {"status": "skipped", "reason": str(e)}
