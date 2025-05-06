# SQLite DB setup and access
import sqlite3
import os
from dotenv import load_dotenv

load_dotenv()
DB_PATH = os.getenv("DB_PATH", "app/data/attempts.db")

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS attempts (
                attempt_id TEXT PRIMARY KEY,
                user_id TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                raw_audio BLOB
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS deepfake_results (
                attempt_id TEXT,
                is_real INTEGER,
                confidence REAL,
                FOREIGN KEY(attempt_id) REFERENCES attempts(attempt_id)
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS denoised_audio (
                attempt_id TEXT,
                audio BLOB,
                FOREIGN KEY(attempt_id) REFERENCES attempts(attempt_id)
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                attempt_id TEXT,
                user_id TEXT,
                embedding BLOB,
                FOREIGN KEY(attempt_id) REFERENCES attempts(attempt_id)
            )
        """)
        conn.commit()

def get_db_connection():
    return sqlite3.connect(DB_PATH)