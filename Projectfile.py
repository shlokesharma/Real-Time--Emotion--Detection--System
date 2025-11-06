"""
Real-Time / Image-Based Emotion Detection System
------------------------------------------------
This program automatically detects if your environment supports webcam.
If webcam is available → runs live emotion detection.
Otherwise → asks you to upload an image and detects emotion from it.

Usage:
  python emotion_detection.py
"""

import cv2
import time
import numpy as np
from deepface import DeepFace
import os

# -----------------------
# CONFIGURATION
# -----------------------
FRAME_RESIZE_FACTOR = 0.6
FACE_DETECTOR_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
ANALYZE_ACTIONS = ["emotion"]
ANALYZE_DETECTOR = "opencv"
CONFIDENCE_THRESHOLD = 0.0


def check_camera():
    """Check if any working camera is available."""
    for i in range(5):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # CAP_DSHOW avoids backend errors on Windows
        if cap.isOpened():
            cap.release()
            return i
        cap.release()
    return None


def run_webcam_mode(camera_index):
    """Run live webcam emotion detection."""
    print(f"✅ Webcam detected at index {camera_index}. Starting real-time emotion detection...")

    face_cascade = cv2.CascadeClassifier(FACE_DETECTOR_PATH)
    if face_cascade.empty():
        raise IOError("❌ Failed to load Haar cascade for face detection. Check OpenCV installation.")

    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise IOError("❌ Could not open webcam. Check permissions or try restarting your system.")

    prev_time = 0
    font = cv2.FONT_HERSHEY_SIMPLEX

    print("🚀 Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Failed to read frame. Exiting...")
                break

            frame_small = cv2.resize(frame, (0, 0), fx=FRAME_RESIZE_FACTOR, fy=FRAME_RESIZE_FACTOR)
            gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

            for (x, y, w, h) in faces:
                x1 = int(x / FRAME_RESIZE_FACTOR)
                y1 = int(y / FRAME_RESIZE_FACTOR)
                x2 = int((x + w) / FRAME_RESIZE_FACTOR)
                y2 = int((y + h) / FRAME_RESIZE_FACTOR)

                face_img = frame[y1:y2, x1:x2].copy()
                if face_img.size == 0:
                    continue

                try:
                    result = DeepFace.analyze(
                        face_img,
                        actions=ANALYZE_ACTIONS,
                        enforce_detection=False,
                        detector_backend=ANALYZE_DETECTOR,
                        prog_bar=False
                    )
                    dominant_emotion = result.get("dominant_emotion", "N/A")
                    emotion_scores = result.get("emotion", {})
                    confidence = emotion_scores.get(dominant_emotion, 0)
                except Exception:
                    dominant_emotion = "error"
                    confidence = 0

                label = f"{dominant_emotion} ({confidence:.1f}%)"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label_y = y1 - 10 if y1 - 10 > 10 else y1 + 10
                cv2.putText(frame, label, (x1, label_y), font, 0.7, (0, 255, 0), 2)

            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time else 0.0
            prev_time = curr_time
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25), font, 0.7, (255, 0, 0), 2)

            cv2.imshow("Real-Time Emotion Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("👋 Exiting...")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Released camera and closed all windows.")


def run_image_mode():
    """Run emotion detection using a static image (for environments without webcam)."""
    print("⚠️ No webcam detected.")
    print("📸 Switching to image mode...")

    # Ask user for image path
    img_path = input("\nEnter path of an image file (example: face.jpg): ").strip()

    if not os.path.exists(img_path):
        print("❌ File not found. Please make sure the path is correct.")
        return

    img = cv2.imread(img_path)
    if img is None:
        print("❌ Failed to load image. Try another file.")
        return

    print("🧠 Analyzing emotions...")
    result = DeepFace.analyze(img, actions=ANALYZE_ACTIONS, enforce_detection=False)

    dominant_emotion = result["dominant_emotion"]
    print(f"\n✅ Dominant Emotion: {dominant_emotion}")
    print("Emotion Scores:")
    for k, v in result["emotion"].items():
        print(f"  {k}: {v:.2f}%")

    # Display result image
    cv2.putText(img, f"Emotion: {dominant_emotion}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("Emotion Detection (Image Mode)", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# -----------------------
# MAIN PROGRAM
# -----------------------
if __name__ == "__main__":
    camera_index = check_camera()
    if camera_index is not None:
        run_webcam_mode(camera_index)
    else:
        run_image_mode()
