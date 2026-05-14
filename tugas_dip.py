import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Konfigurasi Halaman
st.set_page_config(page_title="DIP Dashboard - Sheila", layout="wide")

# CSS Kustom untuk tampilan lebih modern
st.markdown("""
    <style>
    .stApp {
        background: #121212;
        color: white;
    }
    section[data-testid="stSidebar"] {
        background-color: #1e1e1e;
    }
    .stImage {
        border: 2px solid #3498db;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🖼️ Digital Image Processing Dashboard")
st.write("Dibuat untuk memenuhi tugas DIP")

# Sidebar Upload
uploaded_file = st.sidebar.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Membaca gambar dengan benar menggunakan numpy
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8) # Perbaikan di sini: np.uint8
    img_bgr = cv2.imdecode(file_bytes, 1)
    
    # 1. Konversi ke RGB untuk tampilan asli
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 2. Grayscale
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 3. Citra Negatif
    img_neg = 255 - img_rgb

    # 4. Citra Biner
    _, img_bin = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

    # 5. Brightness
    img_bright = cv2.convertScaleAbs(img_rgb, beta=50)

    # 6. Rotasi 45 Derajat
    (h, w) = img_rgb.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), 45, 1.0)
    img_rot = cv2.warpAffine(img_rgb, M, (w, h))

    # Menampilkan hasil dalam grid
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("1. Citra Asli")
        st.image(img_rgb, use_container_width=True)
        st.subheader("4. Citra Biner")
        st.image(img_bin, use_container_width=True)

    with col2:
        st.subheader("2. Grayscale")
        st.image(img_gray, use_container_width=True)
        st.subheader("5. Brightness")
        st.image(img_bright, use_container_width=True)

    with col3:
        st.subheader("3. Citra Negatif")
        st.image(img_neg, use_container_width=True)
        st.subheader("6. Rotasi 45°")
        st.image(img_rot, use_container_width=True)

    # 7. Histogram
    st.divider()
    st.subheader("7. Histogram Citra")
    fig, ax = plt.subplots()
    ax.hist(img_gray.ravel(), 256, [0, 256], color='skyblue')
    st.pyplot(fig)

else:
    st.info("Silakan upload gambar di sidebar sebelah kiri.")