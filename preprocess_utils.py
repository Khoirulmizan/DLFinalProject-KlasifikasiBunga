# File: preprocess_utils.py
#
# Modul ini bertugas untuk mendeteksi objek utama yaitu bunga gaes
# dan membuang background yang tidak relevan secara otomatis.

import cv2
import numpy as np

def smart_crop_function(img):
    """
    Menerima input gambar (numpy array dari Keras),
    Mencari kontur terbesar (diasumsikan bunga),
    Dan melakukan cropping + resizing.
    """
    # Keras mengirimkan gambar dalam format float32 (0-1) atau uint8 (0-255)
    # Kita pastikan formatnya uint8 untuk OpenCV
    if img.dtype == np.float32:
        img_uint8 = (img * 255).astype(np.uint8)
    else:
        img_uint8 = img.astype(np.uint8)

    # 1. Konversi ke Grayscale & Blur untuk menghilangkan noise detail
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 2. Edge Detection (Canny) atau Thresholding
    # Kita gunakan Thresholding adaptif untuk memisahkan objek dari background
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3. Cari Kontur
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # Jika tidak ada kontur ditemukan, kembalikan gambar asli (dinormalisasi)
        return img

    # 4. Ambil kontur terbesar (diasumsikan sebagai bunga)
    c = max(contours, key=cv2.contourArea)
    
    # 5. Dapatkan kotak pembatas (Bounding Box)
    x, y, w, h = cv2.boundingRect(c)

    # Validasi ukuran: Jika kotak terlalu kecil (misal noise), abaikan
    if w < 20 or h < 20:
         return img

    # 6. Crop gambar ke area bunga saja
    cropped = img_uint8[y:y+h, x:x+w]

    # 7. Resize kembali ke ukuran target (misal 128x128) agar seragam masuk model
    # Kita gunakan ukuran input asli dari img
    target_h, target_w = img.shape[:2]
    resized = cv2.resize(cropped, (target_w, target_h))

    # 8. Kembalikan ke format float 0-1 (standar Keras)
    return resized.astype(np.float32) / 255.0