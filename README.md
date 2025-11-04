# Real-Time Emotion Detection System

## Project overview
This application detects human emotions (happy, sad, angry, surprised, neutral, etc.) in real time using a webcam. The system uses OpenCV for face detection and DeepFace for emotion classification with a pretrained deep learning model.

## Features
- Real-time detection with webcam
- Bounding box + emotion label with confidence
- Lightweight and easy to run on CPU (GPU optional for better performance)
- Ready for demo and extension (logging, dashboard, multi-face tracking)

## Requirements
- Python 3.8 - 3.11
- See `requirements.txt`

## Installation
```bash
python -m venv venv
# activate venv
pip install -r requirements.txt
