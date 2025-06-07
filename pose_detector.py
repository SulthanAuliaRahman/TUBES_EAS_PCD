import cv2
import mediapipe as mp
import numpy as np
import time

class PoseDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.counter = 0
        self.stage = None
        self.feedback = "Down"  # Mulai dengan Down
        self.progress = 0
        self.last_count_time = 0
        self.min_time_between_counts = 0.5  # Waktu minimum antar hitungan (detik)

        # Style untuk landmark agar lebih terlihat
        self.landmark_drawing_spec = mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4)
        self.connection_drawing_spec = mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 0), thickness=2)

    def calculate_angle(self, a, b, c):
        """Menghitung sudut antara tiga titik (dalam derajat)."""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle

    def process_frame(self, frame):
        """Memproses frame untuk deteksi pose dan menghitung push-up."""
        # Konversi BGR ke RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            # Gambar landmark dengan style khusus
            self.mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.landmark_drawing_spec,
                connection_drawing_spec=self.connection_drawing_spec
            )

            # Ambil koordinat landmark
            landmarks = results.pose_landmarks.landmark
            shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,
                   landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                    landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y]

            # Hitung sudut siku dan pinggul
            elbow_angle = self.calculate_angle(shoulder, elbow, wrist)
            hip_angle = self.calculate_angle(shoulder, hip, knee)
            body_angle = self.calculate_angle(shoulder, hip, knee)

            # Logika push-up
            current_time = time.time()
            if body_angle > 150 and hip_angle > 150:  # Tubuh mendatar dan pinggul lurus
                if elbow_angle > 160 and self.stage == "Down" and (current_time - self.last_count_time) > self.min_time_between_counts:
                    self.stage = "Up"
                    self.counter += 1
                    self.feedback = "Up"
                    self.last_count_time = current_time
                elif elbow_angle < 80 and self.stage == "Up":
                    self.stage = "Down"
                    self.feedback = "Down"
                elif elbow_angle < 80:
                    self.stage = "Down"
                    self.feedback = "Down"
                else:
                    self.feedback = self.stage if self.stage else "Down"
            else:
                self.feedback = "Down"
                self.stage = None  # Reset stage jika postur tidak benar

            # Progress bar berdasarkan sudut siku (0% di 80°, 100% di 170°)
            self.progress = min(max((elbow_angle - 80) / (170 - 80), 0), 1)

            # Tambahkan teks ke frame
            cv2.putText(image, f"Count: {self.counter}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image, f"Stage: {self.feedback}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image, f"Elbow Angle: {int(elbow_angle)}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(image, f"Body Angle: {int(body_angle)}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        else:
            cv2.putText(image, "No Pose Detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            self.feedback = "Down"
            self.progress = 0
            self.stage = None

        return image, self.counter, self.feedback, self.progress

    def reset_counter(self):
        """Reset counter dan stage."""
        self.counter = 0
        self.stage = None
        self.feedback = "Down"
        self.progress = 0