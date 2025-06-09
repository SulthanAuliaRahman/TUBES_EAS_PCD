import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib
import os
import time

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

def draw_info(image, prediction, count):
    # Draw the predicted gesture and push-up count on the image
    cv2.putText(
        image,
        f"Pose: {prediction}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )
    cv2.putText(
        image,
        f"Push-ups: {count}",
        (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )

def extract_features(landmarks):
    landmark_coords = []
    for landmark in landmarks:
        landmark_coords.extend([landmark.x, landmark.y, landmark.z])
    return np.array([landmark_coords])

def display_pushup_counter_page():
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

    # Streamlit session state for counter and state tracking
    if "pushup_count" not in st.session_state:
        st.session_state.pushup_count = 0
    if "last_pose" not in st.session_state:
        st.session_state.last_pose = None
    if "last_down_time" not in st.session_state:
        st.session_state.last_down_time = None

    # Streamlit UI
    st.subheader("Menghitung Push-up")
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

        # Push-up counting logic
        current_time = time.time()
        if predicted_gesture in ["DOWN", "UP"]:
            if st.session_state.last_pose == "DOWN" and predicted_gesture == "UP":
                if st.session_state.last_down_time is not None:
                    time_diff = current_time - st.session_state.last_down_time
                    if time_diff <= 1.0:  # Check if transition is within 1 second
                        st.session_state.pushup_count += 1
            if predicted_gesture == "DOWN":
                st.session_state.last_down_time = current_time
            st.session_state.last_pose = predicted_gesture
        else:
            # Reset last_down_time if gesture is "Lainya" or no pose detected
            st.session_state.last_down_time = None
            st.session_state.last_pose = None

        # Draw prediction and count on the frame
        draw_info(frame, predicted_gesture, st.session_state.pushup_count)

        # Update UI with the annotated frame
        frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Allow Streamlit to process events
        if st.session_state.get("stop_stream", False):
            break

    cap.release()
    cv2.destroyAllWindows()