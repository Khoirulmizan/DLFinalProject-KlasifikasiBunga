# File: app.py

# Aplikasi web Streamlit untuk klasifikasi bunga.
# Jalankan dengan perintah: streamlit run app.py

import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import json
import os
import random
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
        model = tf.keras.models.load_model('flower_classifier_model.keras') 
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

def preprocess_image(image_data, target_size=(224, 224)): # <-- PERUBAHAN 2
    """
    Mengubah gambar ke format yang sesuai untuk model CUSTOM kita.
    """
    img = Image.open(image_data)
    
    if img.mode != 'RGB':
        img = img.convert('RGB')
        
    img = img.resize(target_size) # <-- Sesuaikan dengan target_size 
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
        # --- PERUBAHAN 1 (Fix Warning) ---
        st.image(input_image, caption="Gambar yang Diunggah", use_container_width=True)

with tab2:
    camera_img = st.camera_input("Arahkan kamera ke bunga:")
    if camera_img is not None:
        input_image = camera_img
        # --- PERUBAHAN 2 (Fix Warning) ---
        st.image(input_image, caption="Gambar dari Kamera", use_container_width=True)

# --- 4. Logika Prediksi dan Tampilan Hasil ---

if input_image is not None and model is not None:
    if st.button("🌸 Klasifikasikan Bunga Ini!", type="primary"):
        with st.spinner("Model sedang menganalisis gambar..."):
            
            # 1. Preprocessing
            processed_img = preprocess_image(input_image)
            
            # 2. Prediksi (mendapatkan array probabilitas)
            # Ambil [0] karena outputnya berbentuk batch (misal: [[0.1, 0.8, ...]])
            prediction = model.predict(processed_img)[0]
            
            # 3. Ambil Top 5
            # argsort mengurutkan dari terkecil, ambil 5 terakhir, balik urutannya
            top_5_indices = np.argsort(prediction)[-5:][::-1]
            top_5_probs = prediction[top_5_indices]
            top_5_class_names = [CLASS_NAMES[i] for i in top_5_indices]

            # 4. Ambil info untuk Prediksi Teratas (Top 1)
            top_1_class_name = top_5_class_names[0]
            top_1_confidence = top_5_probs[0] * 100
            top_1_info = DATA_BUNGA.get(top_1_class_name, None)
            
            st.markdown("---")
            st.header("Hasil Prediksi Model")
            
            if top_1_info:
                # Tampilkan hasil utama
                st.success(f"**Prediksi Teratas:** {top_1_info['nama_umum']} ({top_1_confidence:.2f}%)")
                
                # --- TAMPILKAN GAMBAR ACAK ---
                try:
                    st.subheader("Contoh Gambar dari Dataset")
                    folder_path = os.path.join('Dataset Bunga', top_1_class_name)
                    all_images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    
                    if all_images:
                        random_image_name = random.choice(all_images)
                        sample_image_path = os.path.join(folder_path, random_image_name)
                        st.image(sample_image_path, caption=f"Contoh: {top_1_info['nama_umum']}", use_container_width=True)
                    else:
                        st.warning("Tidak dapat menemukan gambar contoh di dataset.")
                except Exception as e:
                    st.error(f"Error saat memuat gambar contoh: {e}")
                # ---------------------------------------------

                # Tampilkan informasi lengkap untuk Top 1
                st.subheader(f"Detail tentang {top_1_info['nama_umum']}")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Nama Latin:** *{top_1_info['nama_latin']}*")
                    st.markdown(f"**Jenis:** {top_1_info['jenis']}")
                    st.markdown(f"**Kerabat:** {top_1_info['kerabat']}")
                with col2:
                    st.write(top_1_info['deskripsi'])
                
                st.markdown("---")
                
                # --- TAMPILKAN TOP 5 ---
                st.subheader("5 Prediksi Teratas:")
                for i in range(len(top_5_class_names)):
                    class_name = top_5_class_names[i]
                    prob = top_5_probs[i] * 100
                    info = DATA_BUNGA.get(class_name)
                    
                    if info:
                        nama_umum = info['nama_umum']
                        if i == 0:
                            st.write(f"**1. {nama_umum} ({prob:.2f}%)**")
                        else:
                            st.write(f"{i+1}. {nama_umum} ({prob:.2f}%)")
                    else:
                        st.write(f"{i+1}. {class_name} ({prob:.2f}%) - (Info tidak ditemukan)")
                # ------------------------------------

            else:
                # Jika data tidak ditemukan di informasi_bunga.py
                st.error(f"Prediksi: {top_1_class_name} (Confidence: {top_1_confidence:.2f}%)")
                st.warning("Informasi detail untuk bunga ini tidak ditemukan.")

else:
    st.info("Silakan unggah gambar atau gunakan kamera untuk memulai klasifikasi.")