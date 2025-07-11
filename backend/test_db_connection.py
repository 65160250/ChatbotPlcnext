# /backend/test_db_connection.py
import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

DB_URL = os.getenv("DATABASE_URL")

def test_raw_connection():
    print("--- Starting Raw Database Connection Test ---")
    if not DB_URL:
        print("🔥 ERROR: DATABASE_URL environment variable not found!")
        return

    print(f"Connecting to: {DB_URL}")
    conn = None
    try:
        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor()

        # 1. นับจำนวนเอกสารทั้งหมดในตาราง
        cur.execute("SELECT COUNT(*) FROM documents;")
        total_count = cur.fetchone()[0]
        print(f"\n✅ Test 1: Total documents count = {total_count}")

        # 2. ดูว่ามี collection อะไรบ้าง และแต่ละอันมีกี่เอกสาร
        cur.execute("SELECT collection, COUNT(*) FROM documents GROUP BY collection;")
        collections = cur.fetchall()
        print("\n✅ Test 2: Document counts per collection:")
        if not collections:
            print("  -> No collections found.")
        else:
            for collection_name, count in collections:
                print(f"  -> Collection '{collection_name}': {count} documents")

        cur.close()

    except Exception as e:
        print(f"\n🔥 An error occurred: {e}")
    finally:
        if conn:
            conn.close()
            print("\n👋 Connection closed.")

if __name__ == "__main__":
    test_raw_connection()