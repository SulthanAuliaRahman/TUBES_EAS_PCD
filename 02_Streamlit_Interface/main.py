import streamlit as st

# Konfigurasi halaman
st.set_page_config(page_title="Pose Detection App", layout="wide")

# Sidebar untuk navigasi
page = st.sidebar.selectbox(
    "Pilih Halaman",
    ["Home", "Capture"]
)

# Logika untuk menampilkan halaman berdasarkan pilihan
if page == "Home":
    st.title("Selamat Datang di Pose Detection App")
    st.write("Ini adalah halaman utama aplikasi. Pilih halaman dari sidebar untuk melanjutkan.")
    # Placeholder untuk konten Home
    st.write("Konten Home akan ditambahkan di sini.")

elif page == "Capture":
    st.title("Capture Gesture")
    st.write("Halaman untuk menangkap gerakan.")
    # Placeholder untuk konten Capture Gesture
    st.write("Konten Capture Gesture akan ditambahkan di sini.")