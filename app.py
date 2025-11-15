# File: app.py

# Aplikasi web Streamlit untuk klasifikasi bunga.
# Jalankan dengan perintah: streamlit run app.py

import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import json
from io import BytesIO

# --- 1. Muat Aset yang Dibutuhkan ---

# Muat informasi bunga dari file .py
# Kita perlu sedikit trik untuk impor dari file .py
try:
    from informasi_bunga import DATA_BUNGA
except ImportError:
    st.error("Gagal memuat file `informasi_bunga.py`. Pastikan file tersebut ada.")
    # Sediakan data dummy jika gagal agar aplikasi tetap jalan
    DATA_BUNGA = {"bunga1": {"nama_umum": "Error", "deskripsi": "Data tidak ditemukan"}}

# Muat model yang sudah dilatih
# Gunakan @st.cache_resource agar model tidak di-load ulang setiap saat
@st.cache_resource
def load_model():
    try:
        # Ganti nama file model yang dimuat
        model = tf.keras.models.load_model('flower_classifier_scratch.keras') 
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

model = load_model()

# Muat nama-nama kelas
try:
    with open('class_names.json', 'r') as f:
        CLASS_NAMES = json.load(f)
except FileNotFoundError:
    st.error("File `class_names.json` tidak ditemukan. Jalankan `train_model.py` terlebih dahulu.")
    # Sediakan data dummy
    CLASS_NAMES = ["bunga1"]

# --- 2. Fungsi Helper untuk Prapemrosesan Gambar ---

def preprocess_image(image_data, target_size=(150, 150)): # <-- PERUBAHAN 2
    """
    Mengubah gambar ke format yang sesuai untuk model CUSTOM kita.
    """
    img = Image.open(image_data)
    
    if img.mode != 'RGB':
        img = img.convert('RGB')
        
    img = img.resize(target_size) # <-- Sesuaikan dengan target_size (150, 150)
    img_array = np.asarray(img)
    img_array = np.expand_dims(img_array, axis=0) 
    img_array = img_array / 255.0 # Rescale (sesuai training)
    return img_array

# --- 3. Tampilan (UI) Aplikasi Streamlit ---

st.set_page_config(page_title="Klasifikasi Bunga ITERA", layout="wide")
st.title("🌺 Klasifikasi Bunga Kebun Raya ITERA")
st.write("Oleh: Kelompok 2 (Irvan Alfaritzi, Khoirul Mizan A., Syadza Puspadari A.)")
st.markdown("---")

# Pilihan input: Upload Gambar atau Akses Kamera
st.header("Pilih Sumber Gambar")

# Gunakan tab untuk tampilan yang lebih rapi
tab1, tab2 = st.tabs(["Unggah Gambar dari Galeri", "Ambil Gambar dengan Kamera"])

# Inisialisasi variabel input_image
input_image = None

with tab1:
    uploaded_file = st.file_uploader(
        "Pilih file gambar...", 
        type=["jpg", "jpeg", "png"]
    )
    if uploaded_file is not None:
        input_image = BytesIO(uploaded_file.getvalue())
        st.image(input_image, caption="Gambar yang Diunggah", use_column_width=True)

with tab2:
    camera_img = st.camera_input("Arahkan kamera ke bunga:")
    if camera_img is not None:
        input_image = camera_img
        st.image(input_image, caption="Gambar dari Kamera", use_column_width=True)

# --- 4. Logika Prediksi dan Tampilan Hasil ---

if input_image is not None and model is not None:
    # Tombol untuk memicu prediksi
    if st.button("Klasifikasikan Bunga Ini!", type="primary"):
        with st.spinner("Model sedang menganalisis gambar..."):
            
            # 1. Preprocessing
            processed_img = preprocess_image(input_image)
            
            # 2. Prediksi
            prediction = model.predict(processed_img)
            
            # 3. Ambil hasil
            predicted_class_index = np.argmax(prediction)
            predicted_class_name = CLASS_NAMES[predicted_class_index] # e.g., "bunga5"
            confidence = np.max(prediction) * 100
            
            # 4. Ambil informasi bunga dari DATA_BUNGA
            info_bunga = DATA_BUNGA.get(predicted_class_name, None)
            
            st.markdown("---")
            st.header("Hasil Prediksi Model")
            
            if info_bunga:
                st.success(f"**Prediksi:** {info_bunga['nama_umum']} ({confidence:.2f}%)")
                
                # Tampilkan informasi lengkap
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Informasi Bunga")
                    st.markdown(f"**Nama Latin:** *{info_bunga['nama_latin']}*")
                    st.markdown(f"**Jenis:** {info_bunga['jenis']}")
                    st.markdown(f"**Kerabat:** {info_bunga['kerabat']}")
                
                with col2:
                    st.subheader("Deskripsi")
                    st.write(info_bunga['deskripsi'])
            
            else:
                # Jika data tidak ditemukan di informasi_bunga.py
                st.error(f"Prediksi: {predicted_class_name} (Confidence: {confidence:.2f}%)")
                st.warning("Informasi detail untuk bunga ini tidak ditemukan di `informasi_bunga.py`.")

else:
    st.info("Silakan unggah gambar atau gunakan kamera untuk memulai klasifikasi.")