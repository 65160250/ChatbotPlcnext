# retriever.py (เวอร์ชันแก้ไข)
import logging
from typing import List, Any
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from pydantic import Field
from flashrank import Ranker, RerankRequest
import json
from pgvector.psycopg2 import register_vector # <-- เพิ่มการ import ที่สำคัญ

class PostgresVectorRetriever(BaseRetriever):
    """ใช้ Connection Pool และลงทะเบียน Vector Type อย่างถูกต้อง"""
    connection_pool: Any = Field(...)
    embedder: Any = Field(...)
    collection: str = Field(default="plcnext")
    limit: int = Field(default=50) # <-- เพิ่ม limit เป็น 50 ตามคำแนะนำก่อนหน้า

    def _get_relevant_documents(self, query: str) -> List[Document]:
        query_vector = self.embedder.encode(query) # ไม่ต้อง .tolist() แล้ว
        
        conn = self.connection_pool.getconn()
        try:
            register_vector(conn) # <-- **บรรทัดสำคัญ: ลงทะเบียน vector type ทุกครั้งที่เชื่อมต่อ**
            with conn.cursor() as cur:
                sql = """
                SELECT content, metadata, embedding <-> %s as distance
                FROM documents 
                WHERE collection = %s
                ORDER BY embedding <-> %s
                LIMIT %s
                """
                cur.execute(sql, (query_vector, self.collection, query_vector, self.limit))
                rows = cur.fetchall()
            
            documents = []
            for row in rows:
                content = row[0]
                metadata = row[1] if row[1] else {}
                if isinstance(metadata, str):
                    metadata = json.loads(metadata)
                metadata['distance'] = float(row[2])
                documents.append(Document(page_content=content, metadata=metadata))
            
            return documents
        except Exception as e:
            logging.error(f"🔥 Error in PostgresVectorRetriever: {e}")
            return []
        finally:
            self.connection_pool.putconn(conn)


class EnhancedFlashrankRerankRetriever(BaseRetriever):
    """เพิ่ม Domain-specific Boosting"""
    base_retriever: BaseRetriever = Field(...)
    ranker: Ranker = Field(default_factory=lambda: Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="/app/models"))
    top_n: int = Field(default=8)

    def _calculate_domain_boost(self, doc: Document, query: str) -> float:
        boost = 0.0
        content_lower = doc.page_content.lower()

        # เดิม
        plcnext_terms = ["plcnext", "phoenix contact", "gds", "esm", "profinet", "axc f"]
        term_matches = sum(1 for term in plcnext_terms if term in content_lower)
        boost += term_matches * 0.1

        if any(phrase in content_lower for phrase in query.lower().split()):
            boost += 0.3

        # **เพิ่ม block สำหรับ protocol/mode**
        protocol_keywords = [
            "protocol", "mode", "rs-485", "profinet", "ethernet",
            "serial", "communication", "modbus", "tcp", "udp", "interface"
        ]
        protocol_matches = sum(1 for kw in protocol_keywords if kw in content_lower)
        if protocol_matches > 0:
            boost += protocol_matches * 0.5  # ให้คะแนนสูงขึ้นเมื่อเจอ protocol/mode

        return boost

    def _get_relevant_documents(self, query: str) -> List[Document]:
        try:
            candidate_docs = self.base_retriever.get_relevant_documents(query)
            if not candidate_docs: 
                return []

            passages = [{"id": i, "text": doc.page_content, "meta": doc.metadata} for i, doc in enumerate(candidate_docs)]
            rerank_request = RerankRequest(query=query, passages=passages)
            reranked_results = self.ranker.rerank(rerank_request)

            final_docs_with_scores = []
            for result in reranked_results:
                doc = candidate_docs[result["id"]]
                metadata = doc.metadata
                
                # คำนวณ boost score
                boost = 0.0
                if metadata.get("chunk_type") == "golden_qa":
                    boost += 10.0  # ให้คะแนนพิเศษสูงมากสำหรับ Golden Answer
                elif metadata.get("chunk_type") == "spec_pair":
                    boost += 0.2
                boost += self._calculate_domain_boost(doc, query)

                final_score = result["score"] + boost
                final_docs_with_scores.append({"doc": doc, "score": final_score})

            # จัดเรียงใหม่ตามคะแนนสุดท้าย
            final_docs_with_scores.sort(key=lambda x: x["score"], reverse=True)
            
            # ส่งคืนเฉพาะ Document object
            return [item["doc"] for item in final_docs_with_scores[:self.top_n]]
        except Exception as e:
            logging.error(f"🔥 Error in EnhancedFlashrankRerankRetriever: {e}", exc_info=True)
            return []
