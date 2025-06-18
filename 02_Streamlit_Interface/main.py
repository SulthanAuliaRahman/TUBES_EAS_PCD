import streamlit as st
from pushupCounter import display_pushup_counter_page
from squatCounter import display_squat_counter_page

# Konfigurasi halaman
st.set_page_config(page_title="Pose Detection App", layout="wide")

# Sidebar untuk navigasi
page = st.sidebar.selectbox(
    "Pilih Halaman",
    ["Home", "PushUpCounter", "SquatCounter"]
)

# Logika untuk menampilkan halaman berdasarkan pilihan
if page == "Home":
    st.title("Selamat Datang di Pose Detection App")
    st.write("Ini adalah aplikasi untuk menghitung jumlah push-up dan squat.")
    st.write("Pilih halaman dari sidebar untuk melanjutkan:")
    st.write("- **PushUpCounter**: Menghitung jumlah push-up.")
    st.write("- **SquatCounter**: Menghitung jumlah squat.")

elif page == "PushUpCounter":
    st.title("Push-up Counter")
    st.write("Halaman untuk menghitung jumlah push-up.")
    display_pushup_counter_page()

elif page == "SquatCounter":
    st.title("Squat Counter")
    st.write("Halaman untuk menghitung jumlah squat.")
    display_squat_counter_page()