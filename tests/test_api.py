from fastapi.testclient import TestClient
from main import app
import os
import cv2
import numpy as np
from unittest.mock import patch

client = TestClient(app)


def create_dummy_image(filename):
    """Create a simple test image with a bright rectangle (not a real face)."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (150, 150), (255, 255, 255), -1)
    cv2.imwrite(filename, img)


def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "SmartPresence CV Backend is running"}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


@patch("main.get_face_embedding")
@patch("main.save_embedding")
def test_register_student(mock_save, mock_get_embedding):
    # Mock embedding return (128-dim vector to match SFace output)
    mock_get_embedding.return_value = np.random.rand(128).astype(np.float32)

    create_dummy_image("test_face.jpg")
    try:
        with open("test_face.jpg", "rb") as f:
            response = client.post(
                "/register",
                data={"student_id": "12345"},
                files={"file": ("test_face.jpg", f, "image/jpeg")}
            )
        assert response.status_code == 200
        assert "registered successfully" in response.json()["message"]
        mock_save.assert_called_once()
    finally:
        if os.path.exists("test_face.jpg"):
            os.remove("test_face.jpg")


@patch("main.get_face_embedding")
def test_register_no_face(mock_get_embedding):
    """Test registration with an image that has no face."""
    mock_get_embedding.return_value = None

    create_dummy_image("test_noface.jpg")
    try:
        with open("test_noface.jpg", "rb") as f:
            response = client.post(
                "/register",
                data={"student_id": "99999"},
                files={"file": ("test_noface.jpg", f, "image/jpeg")}
            )
        assert response.status_code == 400
        assert "No face found" in response.json()["detail"]
    finally:
        if os.path.exists("test_noface.jpg"):
            os.remove("test_noface.jpg")


@patch("main.get_all_embeddings")
@patch("main.recognize_faces")
def test_mark_attendance(mock_recognize, mock_get_embeddings):
    mock_get_embeddings.return_value = {"12345": np.random.rand(128).astype(np.float32)}
    mock_recognize.return_value = ["12345"]

    create_dummy_image("test_class.jpg")
    try:
        with open("test_class.jpg", "rb") as f:
            response = client.post(
                "/mark-attendance",
                # Fixed: field name must be "files" (plural) to match the API parameter
                files={"files": ("test_class.jpg", f, "image/jpeg")}
            )
        assert response.status_code == 200
        assert response.json()["present_students"] == ["12345"]
    finally:
        if os.path.exists("test_class.jpg"):
            os.remove("test_class.jpg")


@patch("main.get_all_embeddings")
@patch("main.recognize_faces")
def test_mark_attendance_no_matches(mock_recognize, mock_get_embeddings):
    """Test attendance marking when no students are recognized."""
    mock_get_embeddings.return_value = {}
    mock_recognize.return_value = []

    create_dummy_image("test_empty_class.jpg")
    try:
        with open("test_empty_class.jpg", "rb") as f:
            response = client.post(
                "/mark-attendance",
                files={"files": ("test_empty_class.jpg", f, "image/jpeg")}
            )
        assert response.status_code == 200
        assert response.json()["present_students"] == []
    finally:
        if os.path.exists("test_empty_class.jpg"):
            os.remove("test_empty_class.jpg")
