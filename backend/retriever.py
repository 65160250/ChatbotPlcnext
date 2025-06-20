# backend/retriever.py
from typing import List
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from pydantic import Field
from typing import Any
import psycopg2

class PostgresVectorRetriever(BaseRetriever):
    connection_string: str = Field(...)
    table: str = Field(default="documents")
    embedder: Any = Field(...) # รองรับ SentenceTransformer หรือ model อื่น
    collection: str = Field(default="default")

    def _get_relevant_documents(self, query: str) -> List[Document]:
        query_vector = self.embedder.encode(query).tolist()
        with psycopg2.connect(self.connection_string) as conn:
            with conn.cursor() as cur:
                cur.execute(f"""
                    SELECT content FROM {self.table}
                    WHERE collection = %s
                    ORDER BY embedding <-> %s::vector
                    LIMIT 5;
                """, (self.collection, query_vector))
                rows = cur.fetchall()
        return [Document(page_content=row[0]) for row in rows]

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        return self._get_relevant_documents(query)
