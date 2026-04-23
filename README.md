# SmartPresence - CV Backend

The Computer Vision backend for SmartPresence, powered by FastAPI and OpenCV.

## Features
- **Face Registration**: Generates 128-d embeddings for student faces and stores them in PostgreSQL.
- **Face Recognition**: Processes classroom photos to identify present students by comparing embeddings.
- **Health Checks**: Simple endpoint to verify service status.
- **Dockerized**: Ready for production deployment on platforms like Render or Railway.

## CV Models
Uses the following pre-trained ONNX models:
- **YuNet**: High-speed face detection.
- **SFace**: Accurate face recognition and embedding extraction.

## API Endpoints
- `POST /register`: Register a student's face.
- `POST /mark-attendance`: Process images to detect known faces.
- `GET /health`: Basic health monitoring.

## Deployment
This service is designed to run in a Docker container to ensure all OpenCV system dependencies are present.
- **Port**: 8000
- **Database**: PostgreSQL (via `DATABASE_URL` environment variable).
