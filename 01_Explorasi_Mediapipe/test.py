import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib
import os

# Path configuration
MODEL_PATH = "01_Explorasi_Mediapipe/pose_landmarker_full.task"
CLASSIFIER_PATH = "gesture_classifier_cv.pkl"

# MediaPipe setup
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS

def setup_pose_landmarker(model_path):
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO
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

def draw_prediction(image, prediction):
    # Draw the predicted gesture label on the image
    cv2.putText(
        image,
        f"Pose: {prediction}",
        (10, 30),  # Position at top-left corner
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,  # Font scale
        (0, 255, 255),  # Yellow text color
        2,  # Thickness
        cv2.LINE_AA
    )

def extract_features(landmarks):
    landmark_coords = []
    for landmark in landmarks:
        landmark_coords.extend([landmark.x, landmark.y, landmark.z])
    return np.array([landmark_coords])

def display_test_page():
    # Check if classifier model exists
    if not os.path.exists(CLASSIFIER_PATH):
        st.error("Model 'gesture_classifier_cv.pkl' tidak ditemukan. Silakan latih model terlebih dahulu.")
        return

    # Load the classifier
    try:
        classifier = joblib.load(CLASSIFIER_PATH)
    except Exception as e:
        st.error(f"Gagal memuat model: {str(e)}")
        return

    # Streamlit UI
    st.subheader("Identifikasi Gerakan")
    frame_window = st.image([], channels="RGB")

    # Initialize webcam and landmarker
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Kamera tidak tersedia.")
        return

    landmarker = setup_pose_landmarker(MODEL_PATH)

    # Video streaming loop
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Gagal membaca frame dari kamera.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC)) or 0
        result = landmarker.detect_for_video(mp_image, timestamp)

        # Process landmarks and predict gesture
        predicted_gesture = "Tidak ada pose terdeteksi"
        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]
            draw_landmarks(frame, landmarks)
            features = extract_features(landmarks)
            try:
                predicted_gesture = classifier.predict(features)[0]
            except Exception as e:
                predicted_gesture = f"Prediksi gagal: {str(e)}"

        # Draw prediction on the frame
        draw_prediction(frame, predicted_gesture)

        # Update UI with the annotated frame
        frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Allow Streamlit to process events
        if st.session_state.get("stop_stream", False):
            break

    cap.release()
    cv2.destroyAllWindows()