# File: train_model.py
# Jalankan dengan perintah: python train_model.py
# Skrip untuk melatih model CNN.
# Fokus utama: Augmentasi data dan arsitektur custom.

import tensorflow as tf  # noqa: F401
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D # type: ignore
from tensorflow.keras.applications import MobileNetV2 # type: ignore
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping # type: ignore
from sklearn.utils import class_weight
import numpy as np
import os  # noqa: F401
import json
import matplotlib.pyplot as plt

# --- 1. Konfigurasi Parameter ---
DATASET_DIR = 'Dataset Bunga'
# Kita gunakan ukuran gambar yang sedikit lebih kecil agar lebih cepat dilatih
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
NUM_CLASSES = 21 # Sesuai jumlah folder (bunga1 s/d bunga22)
EPOCHS = 80 # Kita butuh lebih banyak epoch untuk belajar dari nol
LEARNING_RATE = 0.001 # Learning rate standar untuk Adam

# --- 2. Prapemrosesan & Augmentasi Data ---
# Untuk data latih, kita akan menggunakan augmentasi untuk mencegah overfitting
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,        # Augmentasi: Rotasi
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
    preprocessing_function=preprocess_input,
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

# 1. Muat Base Model MobileNetV2 sebagai Pengekstrak Fitur
base_model = MobileNetV2(
    input_shape=(*IMG_SIZE, 3),
    include_top=False, # Jangan sertakan classifier ImageNet
    weights='imagenet' # Gunakan bobot ImageNet
)

# 2. Bekukan Base Model karena kita hanya menggunakannya sebagai pengekstrak fitur saja
base_model.trainable = False

# 3. Bangun Model CNN sendiri di atas base model
inputs = base_model.output
x = GlobalAveragePooling2D()(inputs)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(NUM_CLASSES, activation='softmax')(x)

# 4. Gabungkan
model = Model(inputs=base_model.input, outputs=outputs)

# --- 4. Kompilasi dan Pelatihan Model ---
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print(model.summary())

# ReduceLROnPlateau akan memonitor 'val_loss',
# Jika val_loss tidak membaik (stuck) selama 'patience' (3 epoch),
# learning rate akan dikurangi (factor=0.5)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,  # Kurangi LR sebesar 50%
    patience=3,  # Tunggu 3 epoch
    min_lr=0.00001, # Batas LR terkecil
    verbose=1      # Beri tahu kita saat LR diubah
)

# Callback untuk menghentikan training saat mentok
early_stopping = EarlyStopping(
    monitor='val_loss',      # Awasi val_loss
    patience=10,             # Hentikan setelah 10 epoch tidak ada perbaikan
    verbose=1,
    restore_best_weights=True  # Kembalikan bobot terbaik
)

# --- Hitung Class Weights untuk data tidak seimbang ---
print("\n--- Menghitung Class Weights untuk Imbalance ---")

# Dapatkan label kelas untuk setiap sampel di data training
# train_generator.classes akan memberi kita array [0, 0, 1, 1, 1, 2, 3, 3, ...]
training_classes = train_generator.classes

# Hitung bobotnya
class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(training_classes),
    y=training_classes
)

# Ubah menjadi format dictionary yang Keras inginkan
# {class_index: weight}
# {0: 1.2, 1: 0.8, 2: 5.4, ...}
class_weights_dict = dict(enumerate(class_weights))

print("Class weights berhasil dihitung.")

# Mulai pelatihan model dengan class weights
print("\n--- Memulai Pelatihan Model (Versi Stabil + Class Weights) ---")
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    callbacks=[reduce_lr, early_stopping],
    class_weight=class_weights_dict
)

# --- 5. Membuat Plot Training Curve ---
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

# --- 5. Evaluasi dan Penyimpanan Model ---

print("\n--- Evaluasi Model ---")
val_loss, val_accuracy = model.evaluate(validation_generator)
print(f"Akurasi Validasi: {val_accuracy * 100:.2f}%")
print(f"Loss Validasi: {val_loss:.4f}")

# Simpan model
MODEL_SAVE_PATH = 'flower_classifier_model.keras'
model.save(MODEL_SAVE_PATH)
print(f"\nModel telah disimpan di: {MODEL_SAVE_PATH}")

# Simpan nama kelas
with open('class_names.json', 'w') as f:
    json.dump(class_names, f)
print("Nama kelas telah disimpan di: class_names.json")