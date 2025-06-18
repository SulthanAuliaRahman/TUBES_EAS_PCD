import streamlit as st
from capture import display_capture_page
from train import display_train_page
from test import display_test_page
from capture_squat import display_capture_squat_page
from train_squat import display_train_squat_page
from test_squat import display_test_squat_page

# Konfigurasi halaman
st.set_page_config(page_title="Pose Detection App", layout="wide")

# Sidebar untuk navigasi
page = st.sidebar.selectbox(
    "Pilih Halaman",
    ["Home", "Capture Push Up", "Capture Squat", "Train Push Up", "Train Squat", "Test Push Up", "Test Squat"]
)

# Logika untuk menampilkan halaman berdasarkan pilihan
if page == "Home":
    st.title("Selamat Datang di Pose Detection App")
    st.write("Ini adalah aplikasi untuk menangkap, melatih, dan menguji gerakan push-up dan squat.")
    st.write("Pilih halaman dari sidebar untuk melanjutkan:")
    st.write("- **Capture Push Up**: Menangkap gerakan push-up.")
    st.write("- **Capture Squat**: Menangkap gerakan squat.")
    st.write("- **Train Push Up**: Melatih model klasifikasi untuk push-up.")
    st.write("- **Train Squat**: Melatih model klasifikasi untuk squat.")
    st.write("- **Test Push Up**: Menguji model klasifikasi untuk push-up.")
    st.write("- **Test Squat**: Menguji model klasifikasi untuk squat.")

elif page == "Capture Push Up":
    st.title("Capture Push Up")
    st.write("Halaman untuk menangkap gerakan push-up.")
    display_capture_page()

elif page == "Capture Squat":
    st.title("Capture Squat")
    st.write("Halaman untuk menangkap gerakan squat.")
    display_capture_squat_page()

elif page == "Train Push Up":
    st.title("Train Push Up Classifier")
    st.write("Halaman untuk melatih model klasifikasi gerakan push-up.")
    display_train_page()

elif page == "Train Squat":
    st.title("Train Squat Classifier")
    st.write("Halaman untuk melatih model klasifikasi gerakan squat.")
    display_train_squat_page()

elif page == "Test Push Up":
    st.title("Test Push Up Classifier")
    st.write("Halaman untuk mengidentifikasi gerakan push-up menggunakan model yang telah dilatih.")
    display_test_page()

elif page == "Test Squat":
    st.title("Test Squat Classifier")
    st.write("Halaman untuk mengidentifikasi gerakan squat menggunakan model yang telah dilatih.")
    display_test_squat_page()