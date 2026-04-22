import cv2
import numpy as np
import os
import urllib.request

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

YUNET_PATH = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
SFACE_PATH = os.path.join(MODELS_DIR, "face_recognition_sface_2021dec.onnx")

# Lazy-loaded model singletons (avoids blocking import-time downloads)
_face_detector = None
_face_recognizer = None


def _download_model(url, path):
    """Download an ONNX model if it doesn't already exist locally."""
    if not os.path.exists(path):
        print(f"Downloading model {os.path.basename(path)}... This might take a minute.")
        urllib.request.urlretrieve(url, path)
        print(f"Download complete: {os.path.basename(path)}")


def _get_detector():
    """Lazy-load the YuNet face detector."""
    global _face_detector
    if _face_detector is None:
        _download_model(YUNET_URL, YUNET_PATH)
        _face_detector = cv2.FaceDetectorYN_create(
            model=YUNET_PATH,
            config="",
            input_size=(320, 320),  # Will be updated dynamically per image
            score_threshold=0.8,
            nms_threshold=0.3,
            top_k=5000
        )
    return _face_detector


def _get_recognizer():
    """Lazy-load the SFace recognizer."""
    global _face_recognizer
    if _face_recognizer is None:
        _download_model(SFACE_URL, SFACE_PATH)
        _face_recognizer = cv2.FaceRecognizerSF_create(
            model=SFACE_PATH,
            config=""
        )
    return _face_recognizer


def get_face_embedding(image_file):
    """Extract a 128-dim face embedding from the first face found in an image.

    Args:
        image_file: Path to the image file.

    Returns:
        A flat numpy array (128,) of float32, or None if no face is detected.
    """
    image = cv2.imread(image_file)
    if image is None:
        return None

    detector = _get_detector()
    recognizer = _get_recognizer()

    height, width, _ = image.shape
    detector.setInputSize((width, height))

    _, faces = detector.detect(image)
    if faces is None or len(faces) == 0:
        return None

    # Use the first valid face
    face = faces[0]

    # Align and extract feature
    aligned_face = recognizer.alignCrop(image, face)
    embedding = recognizer.feature(aligned_face)
    # Return flat array for database storage
    return embedding.flatten()


def recognize_faces(image_files, known_embeddings):
    """Detect and recognize faces across multiple images.

    Args:
        image_files: List of image file paths to scan.
        known_embeddings: Dict mapping student_id -> numpy embedding array.

    Returns:
        List of recognized student IDs.
    """
    found_students = set()

    if not known_embeddings:
        return list(found_students)

    detector = _get_detector()
    recognizer = _get_recognizer()

    known_ids = list(known_embeddings.keys())
    # Database gives us flat lists/arrays, but face_recognizer expects (1, 128) float32 matrices
    known_encs = [np.array(enc, dtype=np.float32).reshape(1, 128) for enc in known_embeddings.values()]

    for image_file in image_files:
        image = cv2.imread(image_file)
        if image is None:
            continue

        height, width, _ = image.shape
        detector.setInputSize((width, height))

        _, faces = detector.detect(image)
        if faces is None:
            continue

        for face in faces:
            try:
                aligned_face = recognizer.alignCrop(image, face)
                embedding = recognizer.feature(aligned_face)

                best_match_name = None
                # Switch to L2 Distance for much more reliable matching thresholds.
                # OpenCV default is 1.128, but diagnostics show false positives around 1.127. 
                # Tightening to 1.050 to safely exclude strangers while keeping true matches (which score ~0.3 - 0.9).
                best_distance = 1.050

                for idx, known_enc in enumerate(known_encs):
                    dist = recognizer.match(known_enc, embedding, cv2.FaceRecognizerSF_FR_NORM_L2)
                    if dist < best_distance:
                        best_distance = dist
                        best_match_name = known_ids[idx]

                if best_match_name:
                    found_students.add(best_match_name)

            except Exception as e:
                print(f"Error processing face: {e}")

    return list(found_students)
