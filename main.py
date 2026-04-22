from typing import List
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from services.cv_service import get_face_embedding, recognize_faces
from database import init_db, save_embedding, get_all_embeddings
import shutil
import os
import tempfile


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern lifespan handler (replaces deprecated on_event)."""
    init_db()
    yield


app = FastAPI(title="SmartPresence CV Backend", lifespan=lifespan)

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"message": "SmartPresence CV Backend is running"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/register")
async def register_student(student_id: str = Form(...), file: UploadFile = File(...)):
    # Use tempfile for safer temp file handling
    suffix = os.path.splitext(file.filename or ".jpg")[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        temp_filename = tmp.name

    try:
        embedding = get_face_embedding(temp_filename)
        if embedding is None:
            raise HTTPException(status_code=400, detail="No face found in the image")

        save_embedding(student_id, embedding)
        return {"message": f"Student {student_id} registered successfully"}
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

from database import check_embedding_exists, delete_embedding

@app.get("/check-registration/{student_id}")
async def check_registration(student_id: str):
    is_registered = check_embedding_exists(student_id)
    return {"registered": is_registered}

@app.delete("/remove-student/{student_id}")
async def remove_student(student_id: str):
    delete_embedding(student_id)
    return {"message": f"Student {student_id} removed from CV database"}

@app.post("/mark-attendance")
async def mark_attendance(files: List[UploadFile] = File(...)):
    temp_filenames = []

    # Save all uploaded files to temp space
    for file in files:
        suffix = os.path.splitext(file.filename or ".jpg")[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_filenames.append(tmp.name)

    try:
        known_embeddings = get_all_embeddings()
        present_students = recognize_faces(temp_filenames, known_embeddings)
        return {"present_students": present_students}
    finally:
        for temp_filename in temp_filenames:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
