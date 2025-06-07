import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode
import av
import cv2
from pose_detector import PoseDetector

# Konfigurasi WebRTC untuk streaming webcam
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.pose_detector = PoseDetector()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img, counter, feedback, progress = self.pose_detector.process_frame(img)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Antarmuka Streamlit
st.title("Push Up Counter")
st.write("Posisikan kamera di samping tubuh Anda untuk mendeteksi push-up. Pilih sumber input di bawah.")

# Pilih sumber input (webcam atau video)
input_source = st.radio("Pilih sumber input:", ("Webcam", "File Video"))

# State untuk menyimpan counter dan feedback
if "counter" not in st.session_state:
    st.session_state.counter = 0
    st.session_state.feedback = "Down"
    st.session_state.progress = 0

# Placeholder untuk counter, feedback, dan progress
counter_placeholder = st.empty()
feedback_placeholder = st.empty()
progress_placeholder = st.progress(0)

# Proses input berdasarkan pilihan
if input_source == "Webcam":
    try:
        ctx = webrtc_streamer(
            key="pushup-counter",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=VideoProcessor,
            media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
        )
        if ctx and ctx.video_processor:
            counter, feedback, progress = (
                ctx.video_processor.pose_detector.counter,
                ctx.video_processor.pose_detector.feedback,
                ctx.video_processor.pose_detector.progress
            )
            st.session_state.counter = counter
            st.session_state.feedback = feedback
            st.session_state.progress = progress
    except Exception as e:
        st.error(f"Error saat menginisialisasi webcam: {str(e)}")

elif input_source == "File Video":
    uploaded_file = st.file_uploader("Unggah file video (MP4, AVI, dll.)", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        # Simpan file sementara
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.read())
        
        # Baca video
        cap = cv2.VideoCapture("temp_video.mp4")
        if not cap.isOpened():
            st.error("Gagal membuka file video.")
        else:
            stframe = st.empty()
            pose_detector = PoseDetector()
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame, counter, feedback, progress = pose_detector.process_frame(frame)
                stframe.image(frame, channels="BGR")
                st.session_state.counter = counter
                st.session_state.feedback = feedback
                st.session_state.progress = progress
                # Tampilkan counter dan feedback di setiap frame
                counter_placeholder.write(f"Push-Up Count: {st.session_state.counter}")
                feedback_placeholder.write(f"Status: {st.session_state.feedback}")
                progress_placeholder.progress(st.session_state.progress)
                # Tambahkan sedikit delay untuk simulasi real-time
                cv2.waitKey(30)
            cap.release()
    else:
        st.warning("Silakan unggah file video untuk memulai.")

# Tombol reset
if st.button("Reset Counter"):
    if input_source == "Webcam" and ctx and ctx.video_processor:
        ctx.video_processor.pose_detector.reset_counter()
    elif input_source == "File Video":
        pose_detector.reset_counter()
    st.session_state.counter = 0
    st.session_state.feedback = "Down"
    st.session_state.progress = 0

# Tampilkan counter, feedback, dan progress bar
counter_placeholder.write(f"Push-Up Count: {st.session_state.counter}")
feedback_placeholder.write(f"Status: {st.session_state.feedback}")
progress_placeholder.progress(st.session_state.progress)