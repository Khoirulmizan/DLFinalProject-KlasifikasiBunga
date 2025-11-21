# File: train_model.py
#
# Skrip untuk melatih model CNN dari NOL.
# Fokus utama: Augmentasi data dan arsitektur custom.

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, GlobalMaxPooling2D, 
    Dense, Dropout, BatchNormalization, Activation
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.regularizers import l2
from sklearn.utils import class_weight
import numpy as np
import json
import matplotlib.pyplot as plt
import os

# --- 1. Konfigurasi Parameter ---
DATASET_DIR = 'Dataset Bunga'
# Kita gunakan ukuran gambar yang sedikit lebih kecil (lebih cepat dilatih)
IMG_SIZE = (224, 224) 
BATCH_SIZE = 32
NUM_CLASSES = 22 # Sesuai jumlah folder (bunga1 s/d bunga22)
EPOCHS = 50 # Kita butuh lebih banyak epoch untuk belajar dari nol
LEARNING_RATE = 0.001 # Learning rate standar untuk Adam

# --- 2. Prapemrosesan & Augmentasi Data (Senjata Utama Kita) ---
# Ini SANGAT PENTING untuk mencegah overfitting
train_datagen = ImageDataGenerator(
    rescale=1./255,           # Normalisasi piksel
    rotation_range=30,        # Augmentasi: Rotasi
    width_shift_range=0.25,    # Augmentasi: Geser horizontal
    height_shift_range=0.25,   # Augmentasi: Geser vertikal
    shear_range=0.2,          # Augmentasi: 'Gunting'
    zoom_range=0.3,           # Augmentasi: Zoom
    horizontal_flip=True,     # Augmentasi: Balik horizontal
    fill_mode='nearest',
    validation_split=0.2      # Pisahkan 20% data untuk validasi
)

# Untuk data validasi, JANGAN augmentasi. Cukup normalisasi.
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

# --- 3. Membangun Arsitektur Model CNN (Dari Nol) ---
# Ini adalah arsitektur yang kita tentukan sendiri

model = Sequential([
    # Tentukan input shape di layer pertama
    Input(shape=(*IMG_SIZE, 3)),
    
    # Blok Konvolusi 1
    # Layer Konvolusi untuk mencari 32 pola/fitur
    Conv2D(32, (3, 3), padding='same'),
    Activation('relu'),
    MaxPooling2D((2, 2)),
    
    # Blok Konvolusi 2
    # Kita tingkatkan jumlah filter agar model belajar fitur yang lebih kompleks
    Conv2D(64, (3, 3), padding='same'),
    Activation('relu'),
    MaxPooling2D((2, 2)),
    
    # Blok Konvolusi 3
    Conv2D(128, (3, 3), padding='same'),
    Activation('relu'),
    MaxPooling2D((2, 2)),

    # Blok Konvolusi 4 (Opsional, jika butuh lebih dalam)
    Conv2D(128, (3, 3), padding='same'),
    Activation('relu'),
    MaxPooling2D((2, 2)),
    
    # 'Kepala' Model (Classifier)
    GlobalMaxPooling2D(),
    
    # Layer Fully-Connected (Dense)
    Dense(128, kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

# --- 4. Kompilasi ---
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print(model.summary())

# Callbacks
# Patience diperlama karena model kecil butuh waktu belajar dari gambar besar
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True, verbose=1)

# Class Weights
training_classes = train_generator.classes
class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(training_classes),
    y=training_classes
)
class_weights_dict = dict(enumerate(class_weights))

print("\n--- Memulai Pelatihan (High-Res Nano Model) ---")
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    callbacks=[reduce_lr, early_stopping],
    class_weight=class_weights_dict
)

# --- 5. Simpan & Plot ---
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(1, len(loss) + 1)

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend(loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend(loc='upper right')

plt.savefig('training_curves_nano.png')

MODEL_SAVE_PATH = 'flower_classifier_nano.keras'
model.save(MODEL_SAVE_PATH)
print(f"Model disimpan sebagai: {MODEL_SAVE_PATH}")
with open('class_names.json', 'w') as f:
    json.dump(class_names, f)