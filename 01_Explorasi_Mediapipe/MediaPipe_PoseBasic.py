import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Path model .task
model_path = '01_Explorasi_Mediapipe/pose_landmarker_full.task'

# MediaPipe class dan opsi
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Koneksi antar titik pose (skeleton)
POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS

# Fungsi menggambar landmark dan skeleton
def draw_landmarks_and_connections(frame, landmarks):
    image_height, image_width, _ = frame.shape
    for landmark in landmarks:
        x_px = int(landmark.x * image_width)
        y_px = int(landmark.y * image_height)
        cv2.circle(frame, (x_px, y_px), 4, (0, 255, 0), -1)

    for connection in POSE_CONNECTIONS:
        start_idx, end_idx = connection
        if start_idx < len(landmarks) and end_idx < len(landmarks):
            x0 = int(landmarks[start_idx].x * image_width)
            y0 = int(landmarks[start_idx].y * image_height)
            x1 = int(landmarks[end_idx].x * image_width)
            y1 = int(landmarks[end_idx].y * image_height)
            cv2.line(frame, (x0, y0), (x1, y1), (0, 0, 255), 2)

# Konfigurasi PoseLandmarker (mode VIDEO = sinkron per frame)
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO
)

# Buka webcam
cap = cv2.VideoCapture(0)

with PoseLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert ke RGB (MediaPipe butuh RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Konversi ke objek Image MediaPipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Ambil timestamp
        timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))

        # Pakai detect_for_video karena kita pakai VIDEO mode
        result = landmarker.detect_for_video(mp_image, timestamp)

        # Gambar hasil jika ada pose
        if result.pose_landmarks:
            draw_landmarks_and_connections(frame, result.pose_landmarks[0])

        # Tampilkan frame dengan landmark dan skeleton
        cv2.imshow("Pose Detection", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

cap.release()
cv2.destroyAllWindows()
