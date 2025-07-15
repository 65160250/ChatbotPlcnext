# test_retriever.py (เวอร์ชันแก้ไข)
import os
import logging
from retriever import EnhancedFlashrankRerankRetriever, PostgresVectorRetriever
from sentence_transformers import SentenceTransformer
from psycopg2 import pool

# --- Configuration ---
DB_URL = os.getenv("DATABASE_URL")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "BAAI/bge-m3")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def main():
    print("--- Initializing Connections & Models ---")
    db_pool = None
    try:
        db_pool = pool.SimpleConnectionPool(1, 1, dsn=DB_URL)
        embedder = SentenceTransformer(EMBED_MODEL_NAME, cache_folder='/app/models')
        
        base_retriever = PostgresVectorRetriever(
            connection_pool=db_pool,
            embedder=embedder,
            collection="plcnext",
            limit=50
        )

        reranker_retriever = EnhancedFlashrankRerankRetriever(
            base_retriever=base_retriever,
            top_n=5 # แสดง 5 อันดับแรกเพื่อการดีบัก
        )
        print("✅ Retriever initialized.")
        print("-" * 20)

        test_query = "what is PLCnext?"
        print(f"Testing with query: '{test_query}'")

        # เรียกใช้ฟังก์ชันแบบปกติ (ไม่ใช่ await)
        results = reranker_retriever.get_relevant_documents(test_query)

        print("-" * 20)
        print(f"Found {len(results)} relevant document(s) after reranking.")
        print("--- TOP 5 RESULTS ---")

        if not results:
            print("🚨 No documents found.")
        else:
            for i, doc in enumerate(results):
                print(f"\n--- Document {i+1} ---")
                print(f"Content: {doc.page_content[:400]}...")
                print(f"Metadata: {doc.metadata}")
                print("-" * 20)

    except Exception as e:
        logging.error(f"🔥 An error occurred in test_retriever: {e}", exc_info=True)
    finally:
        if db_pool:
            db_pool.closeall()
            print("👋 Database connection pool closed.")

if __name__ == "__main__":
    main()