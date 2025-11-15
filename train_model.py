# File: train_model.py
#
# Skrip untuk melatih model CNN dari NOL.
# Fokus utama: Augmentasi data dan arsitektur custom.

import tensorflow as tf  # noqa: F401
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
import os  # noqa: F401
import json
import matplotlib.pyplot as plt

# --- 1. Konfigurasi Parameter ---
DATASET_DIR = 'Dataset Bunga'
# Kita gunakan ukuran gambar yang sedikit lebih kecil (lebih cepat dilatih)
IMG_SIZE = (150, 150) 
BATCH_SIZE = 8
NUM_CLASSES = 22 # Sesuai jumlah folder (bunga1 s/d bunga22)
EPOCHS = 50 # Kita butuh lebih banyak epoch untuk belajar dari nol
LEARNING_RATE = 0.001 # Learning rate standar untuk Adam

# --- 2. Prapemrosesan & Augmentasi Data ---
# Untuk data latih, kita akan menggunakan augmentasi untuk mencegah overfitting
train_datagen = ImageDataGenerator(
    rescale=1./255,           # Normalisasi piksel
    rotation_range=40,        # Augmentasi: Rotasi
    width_shift_range=0.2,    # Augmentasi: Geser horizontal
    height_shift_range=0.2,   # Augmentasi: Geser vertikal
    shear_range=0.2,          # Augmentasi: 'Gunting'
    zoom_range=0.2,           # Augmentasi: Zoom
    horizontal_flip=True,     # Augmentasi: Balik horizontal
    fill_mode='nearest',
    validation_split=0.2      # Pisahkan 20% data untuk validasi
)

# Untuk data validasi, JANGAN augmentasi. Cukup normalisasi aja.
validation_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# Muat data dari direktori
train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    subset='training'
)

validation_generator = validation_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    subset='validation'
)

# Simpan nama-nama kelas
class_indices = train_generator.class_indices
class_names = list(class_indices.keys())
print(f"Kelas yang ditemukan: {class_names}")

# --- 3. Membangun Arsitektur Model CNN ---
# Ini adalah arsitektur yang kita tentukan sendiri

model = Sequential([
    # Tentukan input shape di layer pertama
    Input(shape=(*IMG_SIZE, 3)),
    
    # Blok Konvolusi 1
    # Layer Konvolusi untuk mencari 32 pola/fitur
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    # Layer Pooling untuk mereduksi ukuran (downsampling)
    MaxPooling2D((2, 2)),
    
    # Blok Konvolusi 2
    # Kita tingkatkan jumlah filter agar model belajar fitur yang lebih kompleks
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    
    # Blok Konvolusi 3
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),

    # Blok Konvolusi 4 (Opsional, jika butuh lebih dalam)
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    
    # 'Kepala' Model (Classifier)
    # Layer Flatten: Mengubah data 2D (matrix) menjadi 1D (vektor)
    Flatten(),
    
    # Layer Fully-Connected (Dense)
    Dense(512, activation='relu'),
    
    # Layer Dropout: Senjata melawan overfitting
    # 'Mematikan' 50% neuron secara acak saat training
    Dropout(0.5),
    
    # Layer Output (Fully-Connected)
    # Harus memiliki neuron sejumlah NUM_CLASSES (22)
    # Aktivasi 'softmax' untuk klasifikasi multi-kelas
    Dense(NUM_CLASSES, activation='softmax')
])

# --- 4. Kompilasi dan Pelatihan Model ---

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Tampilkan arsitektur model yang kita buat
print(model.summary())

# Mulai pelatihan
print("\n--- Memulai Pelatihan Model (Dari Nol) ---")
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS
)

# --- Membuat Plot Training Curve ---
print("\n--- Membuat Grafik Training Curve ---")

# Ambil data dari history pelatihan
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

# Dapatkan jumlah epoch yang sebenarnya berjalan
epochs_ran = len(loss)
epochs_range = range(1, epochs_ran + 1)

# Buat 1 gambar berisi 2 subplot (1 baris, 2 kolom)
plt.figure(figsize=(14, 5))

# Subplot 1: Grafik Akurasi
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

# Subplot 2: Grafik Loss
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# Simpan gambar
plt.savefig('training_curves.png')
print("Grafik telah disimpan sebagai: training_curves.png")

# Tampilkan plot (opsional, bisa di-comment jika berjalan di server)
# plt.show()

# --- 5. Evaluasi dan Penyimpanan Model ---

print("\n--- Evaluasi Model ---")
val_loss, val_accuracy = model.evaluate(validation_generator)
print(f"Akurasi Validasi: {val_accuracy * 100:.2f}%")
print(f"Loss Validasi: {val_loss:.4f}")

# Simpan model
MODEL_SAVE_PATH = 'flower_classifier_scratch.keras'
model.save(MODEL_SAVE_PATH)
print(f"\nModel telah disimpan di: {MODEL_SAVE_PATH}")

# Simpan nama kelas
with open('class_names.json', 'w') as f:
    json.dump(class_names, f)
print("Nama kelas telah disimpan di: class_names.json")