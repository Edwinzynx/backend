import sqlite3
import json
import numpy as np
import os

DB_NAME = os.getenv("DB_PATH", "faces.db")

def init_db():
    # Ensure the directory exists
    db_dir = os.path.dirname(DB_NAME)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)
        
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS students
                 (student_id TEXT PRIMARY KEY, embedding TEXT)''')
    conn.commit()
    conn.close()

def save_embedding(student_id, embedding):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    # Convert numpy array to list then to json string
    embedding_json = json.dumps(embedding.tolist())
    c.execute("INSERT OR REPLACE INTO students (student_id, embedding) VALUES (?, ?)",
              (student_id, embedding_json))
    conn.commit()
    conn.close()

def get_all_embeddings():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT student_id, embedding FROM students")
    rows = c.fetchall()
    conn.close()
    
    embeddings = {}
    for student_id, emb_json in rows:
        embeddings[student_id] = np.array(json.loads(emb_json))
    return embeddings

def check_embedding_exists(student_id: str) -> bool:
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('SELECT 1 FROM students WHERE student_id = ?', (student_id,))
    exists = c.fetchone() is not None
    conn.close()
    return exists

def delete_embedding(student_id: str):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('DELETE FROM students WHERE student_id = ?', (student_id,))
    conn.commit()
    conn.close()
