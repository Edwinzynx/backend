import argparse
import sys
import cv2
import numpy as np
from services.cv_service import get_face_embedding, _get_detector, _get_recognizer

def main():
    parser = argparse.ArgumentParser(description="Test CV Recognition")
    parser.add_argument("--ref", nargs='+', help="Reference images in format name=path", required=True)
    parser.add_argument("--target", help="Target classroom image path", required=True)
    
    args = parser.parse_args()
    
    known_encs = []
    known_names = []
    
    print("--- Loading References ---")
    for ref in args.ref:
        if "=" not in ref:
            print(f"Invalid reference format: {ref}. Use name=path")
            sys.exit(1)
        name, path = ref.split("=", 1)
        emb = get_face_embedding(path)
        if emb is None:
            print(f"Failed to find face in reference image for {name}: {path}")
            sys.exit(1)
        known_encs.append(np.array(emb, dtype=np.float32).reshape(1, 128))
        known_names.append(name)
        print(f"Loaded reference for {name}")
        
    print(f"\n--- Processing Target Image ---")
    print(f"Path: {args.target}")
    image = cv2.imread(args.target)
    if image is None:
        print("Failed to load target image")
        sys.exit(1)
        
    detector = _get_detector()
    recognizer = _get_recognizer()
    
    height, width, _ = image.shape
    detector.setInputSize((width, height))
    
    _, faces = detector.detect(image)
    if faces is None:
        print("No faces detected in target image")
        sys.exit(0)
        
    print(f"Detected {len(faces)} face(s) in target image.\n")
    
    for i, face in enumerate(faces):
        print(f"=== Target Face {i+1} ===")
        aligned_face = recognizer.alignCrop(image, face)
        embedding = recognizer.feature(aligned_face)
        
        for name, known_enc in zip(known_names, known_encs):
            dist_cosine = recognizer.match(known_enc, embedding, cv2.FaceRecognizerSF_FR_COSINE)
            dist_l2 = recognizer.match(known_enc, embedding, cv2.FaceRecognizerSF_FR_NORM_L2)
            
            # Print the raw scores
            print(f"  vs {name}:")
            print(f"    Cosine Distance (Lower is better, Thresh ~0.363): {dist_cosine:.4f}")
            print(f"    L2 Distance (Lower is better, Thresh ~1.128): {dist_l2:.4f}")
        print("")

if __name__ == "__main__":
    main()
