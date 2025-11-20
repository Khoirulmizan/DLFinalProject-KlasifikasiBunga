# File: train_model.py
#
# Arsitektur: Custom ResNet (Built from Scratch)
# Fitur: Smart Cropping + L2 Regularization + Residual Connections

import tensorflow as tf  # noqa: F401
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.layers import ( # type: ignore
    Input, Conv2D, BatchNormalization, Activation, 
    MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, Add
)
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping # type: ignore
from tensorflow.keras.regularizers import l2 # type: ignore
import os  # noqa: F401
import json
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
import numpy as np

# Import fungsi "Smart Crop" kita
try:
    from preprocess_utils import smart_crop_function
except ImportError:
    print("Warning: preprocess_utils.py tidak ditemukan. Menggunakan preprocessing standar.")
    smart_crop_function = None

# --- 1. Konfigurasi Parameter ---
DATASET_DIR = 'Dataset Bunga'
# Ukuran 128 cukup untuk menangkap pola bunga tanpa membebani model
IMG_SIZE = (128, 128) 
BATCH_SIZE = 16
NUM_CLASSES = 22 # Update sesuai info terbaru (22 Kelas)
EPOCHS = 100
LEARNING_RATE = 0.001
REG_FACTOR = 0.001 # Kekuatan L2 Regularization (Mencegah overfitting)

# --- 2. Prapemrosesan & Augmentasi Data ---

# Kita masukkan smart_crop_function ke dalam generator
# Note: preprocessing_function berjalan SEBELUM augmentasi
train_datagen = ImageDataGenerator(
    preprocessing_function=smart_crop_function, # <-- FOKUS OBJEK DI SINI
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True, # Bunga bisa dilihat dari atas
    fill_mode='nearest',
    validation_split=0.2
)

validation_datagen = ImageDataGenerator(
    preprocessing_function=smart_crop_function,
    validation_split=0.2
)

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

class_indices = train_generator.class_indices
class_names = list(class_indices.keys())
print(f"Kelas yang ditemukan ({len(class_names)}): {class_names}")

# --- 3. Membangun Arsitektur Model (CUSTOM RESNET) ---

def residual_block(x, filters, kernel_size=3, stride=1):
    """
    Membangun satu blok residual.
    Konsep: Output = Conv(Input) + Input
    Ini membantu model belajar fitur kompleks tanpa lupa fitur dasar.
    """
    shortcut = x
    
    # Layer Konvolusi Utama
    x = Conv2D(filters, kernel_size, strides=stride, padding='same', 
               kernel_regularizer=l2(REG_FACTOR))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(filters, kernel_size, padding='same', 
               kernel_regularizer=l2(REG_FACTOR))(x)
    x = BatchNormalization()(x)
    
    # Penyesuaian Shortcut (jika dimensi berubah karena stride atau filter beda)
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, (1, 1), strides=stride, padding='same',
                          kernel_regularizer=l2(REG_FACTOR))(shortcut)
        shortcut = BatchNormalization()(shortcut)
    
    # KUNCI RESNET: Tambahkan input asli ke hasil konvolusi
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

# Definisi Model Menggunakan Functional API
inputs = Input(shape=(*IMG_SIZE, 3))

# -- Stem (Batang Awal) --
x = Conv2D(32, (7, 7), strides=2, padding='same', kernel_regularizer=l2(REG_FACTOR))(inputs)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((3, 3), strides=2, padding='same')(x)

# -- Body (Blok Residual) --
# Kita tumpuk blok ini untuk mengekstrak fitur dalam (bentuk, tekstur)
x = residual_block(x, 64)
x = residual_block(x, 64)
x = residual_block(x, 128, stride=2) # Downsample
x = residual_block(x, 128)
x = residual_block(x, 256, stride=2) # Downsample
x = residual_block(x, 256)

# -- Head (Klasifikasi) --
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x) # Dropout tinggi karena data sedikit
x = Dense(NUM_CLASSES, activation='softmax', kernel_regularizer=l2(REG_FACTOR))(x)

model = Model(inputs=inputs, outputs=x)

# --- 4. Kompilasi dan Pelatihan ---

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print(model.summary())

# Callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)

# Class Weights (Sangat penting untuk 40 gambar/kelas)
training_classes = train_generator.classes
class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(training_classes),
    y=training_classes
)
class_weights_dict = dict(enumerate(class_weights))

print("\n--- Memulai Pelatihan Custom ResNet (From Scratch) ---")
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    callbacks=[reduce_lr, early_stopping],
    class_weight=class_weights_dict
)

# --- 5. Simpan Model & Plotting ---
# (Sama seperti sebelumnya)
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
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.savefig('training_curves_resnet.png')

MODEL_SAVE_PATH = 'flower_classifier_resnet.keras'
model.save(MODEL_SAVE_PATH)
print(f"\nModel ResNet disimpan di: {MODEL_SAVE_PATH}")
with open('class_names.json', 'w') as f:
    json.dump(class_names, f)