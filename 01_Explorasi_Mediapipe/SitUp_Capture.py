import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import json
import os
from datetime import datetime
from PIL import Image

# Path configuration
MODEL_PATH = "pose_landmarker_full.task"
SAVE_DIR = "Gesture_SitUp"
JSON_FILE = os.path.join(SAVE_DIR, "situp_gesture.json")

# MediaPipe setup
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS

def setup_pose_landmarker(model_path):
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

def process_uploaded_image(image, landmarker):
    # Convert PIL image to OpenCV format
    image_np = np.array(image)
    if image_np.shape[2] == 4:  # Handle RGBA images
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
    # No conversion needed for RGB images (already in correct format)

    # Create MediaPipe image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)

    # Detect pose
    result = landmarker.detect(mp_image)

    # Draw landmarks if detected
    landmarks = None
    if result.pose_landmarks:
        landmarks = result.pose_landmarks[0]
        draw_landmarks(image_np, landmarks)

    return image_np, landmarks

def save_image_data(image_np, landmarks, label, save_path):
    # Save image
    cv2.imwrite(save_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

    # Save landmark data to JSON
    if landmarks is not None:
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
        return True
    return False

def display_situp_page():
    os.makedirs(SAVE_DIR, exist_ok=True)
    if not os.path.exists(JSON_FILE):
        with open(JSON_FILE, "w") as f:
            json.dump([], f)

    # Streamlit UI
    st.subheader("Upload Sit-up Pose Images")
    label = st.selectbox("Pilih Label Pose:", ["START", "UP", "Other"])
    uploaded_files = st.file_uploader("Pilih Gambar", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    process_btn = st.button("Proses Gambar")
    
    if uploaded_files:
        # Display uploaded images
        st.write("Gambar yang diunggah:")
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file)
            st.image(image, channels="RGB", caption=f"Uploaded: {uploaded_file.name}")

        if process_btn:
            # Initialize landmarker
            landmarker = setup_pose_landmarker(MODEL_PATH)

            # Process each uploaded image
            st.write("Hasil pemrosesan:")
            for uploaded_file in uploaded_files:
                image = Image.open(uploaded_file)
                processed_image, landmarks = process_uploaded_image(image, landmarker)

                # Display processed image
                st.image(processed_image, channels="RGB", caption=f"Processed: {uploaded_file.name}")

                # Save data
                filename = f"{label}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                save_path = os.path.join(SAVE_DIR, filename)
                if save_image_data(processed_image, landmarks, label, save_path):
                    st.success(f"✅ Gambar disimpan: {filename} dengan label '{label}'")
                else:
                    st.warning(f"❌ Tidak ada pose terdeteksi dalam gambar: {uploaded_file.name}")