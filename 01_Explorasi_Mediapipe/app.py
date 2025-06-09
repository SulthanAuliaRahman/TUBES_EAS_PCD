import streamlit as st
from capture import display_capture_page
from train import display_train_page
from test import display_test_page

# Konfigurasi halaman
st.set_page_config(page_title="Pose Detection App", layout="wide")

# Sidebar untuk navigasi
page = st.sidebar.selectbox(
    "Pilih Halaman",
    ["Home", "Capture", "Train", "Test"]
)

# Logika untuk menampilkan halaman berdasarkan pilihan
if page == "Home":
    st.title("Selamat Datang di Pose Detection App")
    st.write("Ini adalah halaman utama aplikasi. Pilih halaman dari sidebar untuk melanjutkan.")
    st.write("Konten Home akan ditambahkan di sini.")

elif page == "Capture":
    st.title("Capture Gesture")
    st.write("Halaman untuk menangkap gerakan.")
    display_capture_page()

elif page == "Train":
    st.title("Train Gesture Classifier")
    st.write("Halaman untuk melatih model klasifikasi gerakan.")
    display_train_page()

elif page == "Test":
    st.title("Test Gesture Classifier")
    st.write("Halaman untuk mengidentifikasi gerakan menggunakan model yang telah dilatih.")
    display_test_page()