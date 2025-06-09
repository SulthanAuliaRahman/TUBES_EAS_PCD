import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import json
import os
from datetime import datetime
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Path model
MODEL_PATH = "01_Explorasi_Mediapipe/pose_landmarker_full.task"
SAVE_DIR = "Gesture_pushUP"
JSON_FILE = os.path.join(SAVE_DIR, "gesture.json")

# Setup
os.makedirs(SAVE_DIR, exist_ok=True)
if not os.path.exists(JSON_FILE):
    with open(JSON_FILE, "w") as f:
        json.dump([], f)

# Streamlit session state init
if "capture" not in st.session_state:
    st.session_state.capture = False
if "frame" not in st.session_state:
    st.session_state.frame = None
if "landmarks" not in st.session_state:
    st.session_state.landmarks = None

# MediaPipe setup
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS

# Fungsi gambar landmark + koneksi
def draw_landmarks(image, landmarks):
    image_height, image_width, _ = image.shape
    for landmark in landmarks:
        cx = int(landmark.x * image_width)
        cy = int(landmark.y * image_height)
        cv2.circle(image, (cx, cy), 4, (0, 255, 0), -1)
    for connection in POSE_CONNECTIONS:
        start_idx, end_idx = connection
        if start_idx < len(landmarks) and end_idx < len(landmarks):
            x0 = int(landmarks[start_idx].x * image_width)
            y0 = int(landmarks[start_idx].y * image_height)
            x1 = int(landmarks[end_idx].x * image_width)
            y1 = int(landmarks[end_idx].y * image_height)
            cv2.line(image, (x0, y0), (x1, y1), (255, 0, 0), 2)

# UI
st.title("Pose Dataset Creator")
label = st.selectbox("Pilih Label Pose:", ["DOWN", "UP", "Lainya"])
capture_btn = st.button("📸 Capture")
frame_window = st.image([])

# Mulai Webcam
cap = cv2.VideoCapture(0)

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO
)

with PoseLandmarker.create_from_options(options) as landmarker:
    ret, frame = cap.read()
    if ret:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        result = landmarker.detect_for_video(mp_image, timestamp)

        # Simpan ke state
        st.session_state.frame = frame.copy()
        if result.pose_landmarks:
            st.session_state.landmarks = result.pose_landmarks[0]
            draw_landmarks(frame, result.pose_landmarks[0])

        frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    else:
        st.error("Kamera tidak tersedia.")
        cap.release()

# Handle tombol capture
if capture_btn and st.session_state.landmarks is not None:
    frame = st.session_state.frame
    landmarks = st.session_state.landmarks

    filename = f"{label}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
    save_path = os.path.join(SAVE_DIR, filename)
    cv2.imwrite(save_path, frame)

    landmark_data = [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in landmarks]

    with open(JSON_FILE, "r") as f:
        data = json.load(f)

    data.append({
        "name": label,
        "landmarks": landmark_data,
        "image_path": save_path
    })

    with open(JSON_FILE, "w") as f:
        json.dump(data, f, indent=2)

    st.success(f"✅ Disimpan: {filename} dengan label '{label}'")
elif capture_btn:
    st.warning("❌ Tidak ada pose terdeteksi saat capture.")

cap.release()
cv2.destroyAllWindows()
