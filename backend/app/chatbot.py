

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def preprocess_query(query: str) -> str:
    """...ตาม main.py เดิม..."""
    abbreviations = {
        "plc": "PLCnext", 
        "hmi": "Human Machine Interface", 
        "profinet": "PROFINET",
        "i/o": "input output", 
        "gds": "Global Data Space", 
        "esm": "Execution and Synchronization Manager"
    }
    import re
    processed_query = query.lower()
    for abbr, full_form in abbreviations.items():
        pattern = r'\b' + re.escape(abbr) + r'\b'
        processed_query = re.sub(pattern, full_form, processed_query)
    return processed_query if processed_query != query.lower() else query

def build_enhanced_prompt() -> PromptTemplate:
    """...ตาม main.py เดิม..."""
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

def log_query_performance(query: str, response: str, retrieval_time: float, total_time: float, context_count: int):
    import logging
    logging.info(
        f"📊 Query Performance: "
        f"Query: '{query[:50]}...' | "
        f"Response Length: {len(response)} | "
        f"Context Count: {context_count} | "
        f"Retrieval Time: {retrieval_time:.2f}s | "
        f"Total Time: {total_time:.2f}s"
    )

def answer_question(
    question: str,
    db_pool,
    llm,
    embedder,
    collection: str,
    retriever_class,
    reranker_class
) -> dict:
    """
    ประมวลผลคำถามทั้งหมด: preprocess, retrieve, rerank, compose prompt, infer, log, return answer
    - ใช้ retriever, reranker, LLM, embedder, collection จาก main.py
    """
    import time
    # 1. Validate and preprocess query
    processed_msg = preprocess_query(question.strip())
    if not processed_msg:
        return {
            "reply": "Message cannot be empty.",
            "processing_time": 0,
            "retrieval_time": 0,
            "context_count": 0
        }

    start_time = time.perf_counter()
    retrieval_start = time.perf_counter()

    # 2. Build retriever chain
    base_retriever = retriever_class(
        connection_pool=db_pool,
        embedder=embedder,
        collection=collection,
    )
    reranker_retriever = reranker_class(
        base_retriever=base_retriever
    )
    prompt = build_enhanced_prompt()
    rag_chain = (
        {"context": reranker_retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    # 3. Generate response
    response_text = rag_chain.invoke(processed_msg)
    retrieval_time = time.perf_counter() - retrieval_start
    total_time = time.perf_counter() - start_time

    # 4. Context count
    try:
        context_docs = reranker_retriever.get_relevant_documents(processed_msg)
        context_count = len(context_docs)
    except:
        context_count = 0

    # 5. Log
    log_query_performance(processed_msg, response_text, retrieval_time, total_time, context_count)

    return {
        "reply": response_text,
        "processing_time": total_time,
        "retrieval_time": retrieval_time,
        "context_count": context_count
    }
