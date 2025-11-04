"""
Real-Time Emotion Detection System
Usage: python emotion_realtime.py
Press 'q' to quit the webcam window.
"""

import cv2
from deepface import DeepFace
import time
import numpy as np

# -----------------------
# CONFIG
# -----------------------
CAMERA_INDEX = 0             # default webcam
FRAME_RESIZE_FACTOR = 0.6    # reduce frame size to speed up inference
FACE_DETECTOR_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
ANALYZE_DETECTOR = "opencv"  # DeepFace detector backend for alignment; can be 'mtcnn', 'opencv', 'ssd', etc.
ANALYZE_ACTIONS = ["emotion"] # only emotion analysis
CONFIDENCE_THRESHOLD = 0.0   # show all predictions (DeepFace returns details)

# -----------------------
# Load face detector
# -----------------------
face_cascade = cv2.CascadeClassifier(FACE_DETECTOR_PATH)
if face_cascade.empty():
    raise IOError("Failed to load Haar cascade for face detection. Check OpenCV installation.")

# -----------------------
# Start video capture
# -----------------------
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise IOError("Cannot open webcam. Check camera index or permissions.")

prev_time = 0
font = cv2.FONT_HERSHEY_SIMPLEX

print("Starting Real-Time Emotion Detection. Press 'q' to quit.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from camera. Exiting.")
            break

        # Resize frame for speed
        frame_small = cv2.resize(frame, (0, 0), fx=FRAME_RESIZE_FACTOR, fy=FRAME_RESIZE_FACTOR)
        gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)

        # Detect faces (returns coordinates wrt resized frame)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        # For each face region, run emotion detection using DeepFace (on a copy of the original frame)
        for (x, y, w, h) in faces:
            # map coords back to original frame size
            x1 = int(x / FRAME_RESIZE_FACTOR)
            y1 = int(y / FRAME_RESIZE_FACTOR)
            x2 = int((x + w) / FRAME_RESIZE_FACTOR)
            y2 = int((y + h) / FRAME_RESIZE_FACTOR)

            # Extract face ROI from original frame for better quality
            face_img = frame[y1:y2, x1:x2].copy()
            if face_img.size == 0:
                continue

            # DeepFace analyze (returns dict)
            try:
                result = DeepFace.analyze(face_img, actions=ANALYZE_ACTIONS, enforce_detection=False, detector_backend=ANALYZE_DETECTOR)
                # result example: {'emotion': {'angry':0.1,'disgust':0.0,'fear':0.0,'happy':98.2,...}, 'dominant_emotion': 'happy', ...}
                dominant_emotion = result.get("dominant_emotion", "N/A")
                emotion_scores = result.get("emotion", {})
                confidence = emotion_scores.get(dominant_emotion, None)
            except Exception as e:
                # if deepface fails, skip
                dominant_emotion = "error"
                confidence = None

            # Draw rectangle and label on original frame
            label = dominant_emotion
            if confidence is not None:
                label = f"{dominant_emotion} ({confidence:.1f}%)"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # calculate label position
            label_y = y1 - 10 if y1 - 10 > 10 else y1 + 10
            cv2.putText(frame, label, (x1, label_y), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        # FPS calculation
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0.0
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25), font, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

        # Display
        cv2.imshow("Real-Time Emotion Detection", frame)

        # quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Released camera and closed all windows.")
