import streamlit as st
from pushupCounter import display_pushup_counter_page

# Konfigurasi halaman
st.set_page_config(page_title="Pose Detection App", layout="wide")

# Sidebar untuk navigasi
page = st.sidebar.selectbox(
    "Pilih Halaman",
    ["Home", "PushUpCounter"]
)

# Logika untuk menampilkan halaman berdasarkan pilihan
if page == "Home":
    st.title("Selamat Datang di Pose Detection App")
    st.write("Ini adalah halaman utama aplikasi. Pilih halaman dari sidebar untuk melanjutkan.")
    # Placeholder untuk konten Home
    st.write("Konten Home akan ditambahkan di sini.")

elif page == "PushUpCounter":
    st.title("Push-up Counter")
    st.write("Halaman untuk menghitung jumlah push-up.")
    display_pushup_counter_page()