import psycopg2
import json
import numpy as np
import os
from psycopg2.extras import execute_values

DATABASE_URL = os.getenv("DATABASE_URL")

def get_connection():
    return psycopg2.connect(DATABASE_URL)

def init_db():
    conn = get_connection()
    cur = conn.cursor()
    # Create table for face embeddings if it doesn't exist
    cur.execute('''
        CREATE TABLE IF NOT EXISTS face_embeddings (
            student_id TEXT PRIMARY KEY,
            embedding TEXT NOT NULL
        )
    ''')
    conn.commit()
    cur.close()
    conn.close()

def save_embedding(student_id, embedding):
    conn = get_connection()
    cur = conn.cursor()
    embedding_json = json.dumps(embedding.tolist())
    cur.execute("""
        INSERT INTO face_embeddings (student_id, embedding)
        VALUES (%s, %s)
        ON CONFLICT (student_id) DO UPDATE SET embedding = EXCLUDED.embedding
    """, (student_id, embedding_json))
    conn.commit()
    cur.close()
    conn.close()

def get_all_embeddings():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT student_id, embedding FROM face_embeddings")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    
    embeddings = {}
    for student_id, emb_json in rows:
        embeddings[student_id] = np.array(json.loads(emb_json))
    return embeddings

def check_embedding_exists(student_id: str) -> bool:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM face_embeddings WHERE student_id = %s", (student_id,))
    exists = cur.fetchone() is not None
    cur.close()
    conn.close()
    return exists

def delete_embedding(student_id: str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM face_embeddings WHERE student_id = %s", (student_id,))
    conn.commit()
    cur.close()
    conn.close()
