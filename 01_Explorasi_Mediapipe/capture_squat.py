import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import json
import os
from datetime import datetime
from PIL import Image
import io

# Path configuration
MODEL_PATH = os.path.join(".", "pose_landmarker_full.task")
SAVE_DIR = os.path.join(".", "Gesture_squat")
JSON_FILE = os.path.join(SAVE_DIR, "gesture.json")

# MediaPipe setup
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS

def setup_pose_landmarker(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file tidak ditemukan di: {os.path.abspath(model_path)}")
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE
    )
    return PoseLandmarker.create_from_options(options)

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

def process_image(image, landmarker):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    result = landmarker.detect(mp_image)
    return result.pose_landmarks

def display_capture_squat_page():
    os.makedirs(SAVE_DIR, exist_ok=True)
    if not os.path.exists(JSON_FILE):
        with open(JSON_FILE, "w") as f:
            json.dump([], f)

    # Streamlit session state initialization
    if "frame" not in st.session_state:
        st.session_state.frame = None
    if "landmarks" not in st.session_state:
        st.session_state.landmarks = None
    if "capture_triggered" not in st.session_state:
        st.session_state.capture_triggered = False

    # Streamlit UI
    st.subheader("Capture Gerakan Squat")
    mode = st.selectbox("Pilih Mode Capture:", ["Webcam", "Upload Foto"])
    label = st.selectbox("Pilih Label Pose:", ["DOWN", "UP", "Lainya"])
    frame_window = st.image([], channels="RGB")

    landmarker = setup_pose_landmarker(MODEL_PATH)

    if mode == "Webcam":
        capture_btn = st.button("Capture", key="webcam_capture")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Kamera tidak tersedia.")
            return

        # Set webcam resolution to 1280x720
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Video streaming loop
        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Gagal membaca frame dari kamera.")
                break

            # Process frame for landmarks
            landmarks = process_image(frame, landmarker)

            # Save to session state
            st.session_state.frame = frame.copy()
            st.session_state.landmarks = landmarks[0] if landmarks else None

            # Draw landmarks if detected
            if landmarks:
                draw_landmarks(frame, landmarks[0])

            frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Handle capture button
            if capture_btn or st.session_state.capture_triggered:
                if st.session_state.landmarks is not None:
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
                else:
                    st.warning("❌ Tidak ada pose terdeteksi saat capture.")

                st.session_state.capture_triggered = False
                break

            if capture_btn:
                st.session_state.capture_triggered = True
                break

        cap.release()
        cv2.destroyAllWindows()

    elif mode == "Upload Foto":
        uploaded_file = st.file_uploader("Unggah Gambar (JPG/PNG)", type=["jpg", "png"])
        capture_btn = st.button("Proses Gambar", key="upload_capture")

        if uploaded_file and capture_btn:
            # Read uploaded image
            image = Image.open(uploaded_file)
            image_np = np.array(image)
            if image_np.shape[2] == 4:  # Convert RGBA to RGB if needed
                image_np = image_np[:, :, :3]
            frame = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            # Process image for landmarks
            landmarks = process_image(frame, landmarker)

            # Save to session state
            st.session_state.frame = frame.copy()
            st.session_state.landmarks = landmarks[0] if landmarks else None

            # Draw landmarks if detected
            if landmarks:
                draw_landmarks(frame, landmarks[0])

            frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if st.session_state.landmarks is not None:
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
            else:
                st.warning("❌ Tidak ada pose terdeteksi pada gambar yang diunggah.")