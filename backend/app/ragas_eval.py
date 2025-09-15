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
    - หา block "Question: ...\\nAnswer: ..." ที่ 'คล้าย' กับคำถาม
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
    ใช้ OpenAI เป็นกรรมการได้ พร้อมอนุญาตให้ EVAL_EMBED_MODEL เป็น sentence-transformers/*
    - ถ้ามี OPENAI_API_KEY → judge=OpenAI (เช่น gpt-4o-mini)
    - ฝั่ง Embeddings:
        * ถ้า EVAL_EMBED_MODEL เริ่มด้วย 'sentence-transformers' → ใช้ HuggingFaceEmbeddings
        * ถ้าไม่ใช่ และมี OPENAI_API_KEY → ใช้ OpenAIEmbeddings
        * ถ้าไม่ใช่ และไม่มี OPENAI_API_KEY → ใช้ OllamaEmbeddings
    - ถ้ามี ground_truth (เด้งจาก contexts) จะคำนวณครบ 4 metric มิฉะนั้นคิดเฉพาะ faithfulness/answer_relevancy
    - เพิ่ม fallback ให้ context_precision ถ้า RAGAS คืน None/NaN
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            context_precision, faithfulness, answer_relevancy, context_recall
        )

        use_openai = bool(os.getenv("OPENAI_API_KEY"))

        # --- เลือก LLM (กรรมการ) ---
        if use_openai:
            from langchain_openai import ChatOpenAI
            judge = ChatOpenAI(
                model=os.getenv("RAGAS_JUDGE_MODEL", "gpt-4o-mini"),
                temperature=0
            )
            judge_name = "openai"
        else:
            from langchain_community.chat_models import ChatOllama
            judge = ChatOllama(
                model=os.getenv("OLLAMA_MODEL", "llama3.2"),
                base_url=os.getenv("OLLAMA_BASE_URL", "http://ollama:11434"),
            )
            judge_name = "ollama"

        # --- เลือก Embeddings ตาม EVAL_EMBED_MODEL ---
        em_name = os.getenv("EVAL_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        if em_name.startswith("sentence-transformers"):
            from langchain_community.embeddings import HuggingFaceEmbeddings
            embeddings = HuggingFaceEmbeddings(
                model_name=em_name,
                encode_kwargs={"normalize_embeddings": True}
            )
        else:
            if use_openai:
                from langchain_openai import OpenAIEmbeddings
                embeddings = OpenAIEmbeddings(model=em_name)
            else:
                from langchain_community.embeddings import OllamaEmbeddings
                embeddings = OllamaEmbeddings(
                    model=os.getenv("OLLAMA_EMBED_MODEL", os.getenv("OLLAMA_MODEL", "llama3.2")),
                    base_url=os.getenv("OLLAMA_BASE_URL", "http://ollama:11434"),
                )

        # --- เตรียม contexts (dedup + จำกัด top-k เพื่อความเร็ว) ---
        topk = int(os.getenv("RAGAS_EVAL_TOPK", "6"))  # ตั้งค่าได้ใน env
        seen = set()
        contexts_clean: List[str] = []
        for c in contexts or []:
            s = (c or "").strip()
            if not s or s in seen:
                continue
            seen.add(s)
            contexts_clean.append(s)
            if len(contexts_clean) >= topk:
                break

        # --- สร้าง Dataset ---
        gt = _guess_ground_truth(question, contexts_clean)
        has_gt = bool(gt and gt.strip())
        ds = Dataset.from_dict({
            "question": [question or ""],
            "contexts": [contexts_clean],
            "answer":   [answer or ""],
            "ground_truth": [gt or ""]
        })

        # --- เลือก metrics ---
        metrics = [faithfulness, answer_relevancy]
        if has_gt:
            metrics += [context_precision, context_recall]

        result = evaluate(ds, metrics=metrics, llm=judge, embeddings=embeddings)
        scores = _extract_scores_dataframe_like(result)

        # เติม key ให้ครบ และตั้ง None ให้ metric ที่ไม่ได้คำนวณ
        for k in ["context_precision", "faithfulness", "answer_relevancy", "context_recall"]:
            if k not in scores or (scores[k] is None and (k in ["context_precision", "context_recall"] and not has_gt)):
                scores[k] = None

        # ---- Fallback for context_precision (เมื่อ RAGAS คืน None/NaN) ----
        try:
            import math
            import numpy as np

            def _is_bad(x):
                return (x is None) or (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))

            if has_gt and _is_bad(scores.get("context_precision")) and contexts_clean:
                # ใช้ embeddings เดิม (ไม่ว่า OpenAI/HF/Ollama) คำนวณ cosine แบบทั่วไป
                gt_vec = embeddings.embed_query(gt)
                ctx_vecs = embeddings.embed_documents(contexts_clean)

                def _cosine(a, b):
                    a = np.array(a, dtype=float); b = np.array(b, dtype=float)
                    an = np.linalg.norm(a); bn = np.linalg.norm(b)
                    if an == 0 or bn == 0:
                        return 0.0
                    return float(np.dot(a, b) / (an * bn))

                sims = [_cosine(gt_vec, v) for v in ctx_vecs]
                threshold = float(os.getenv("PRECISION_FALLBACK_THRESHOLD", "0.40"))
                rel = sum(1 for s in sims if s >= threshold)
                total = len(ctx_vecs)
                scores["context_precision"] = (rel / total) if total > 0 else None
        except Exception:
            # ถ้าคำนวณ fallback มีปัญหา ให้คงค่า None ไว้
            pass

        return {
            "status": "ok",
            "scores": scores,
            "judge": judge_name,
            "has_ground_truth": has_gt
        }

    except Exception as e:
        return {"status": "skipped", "reason": str(e)}
