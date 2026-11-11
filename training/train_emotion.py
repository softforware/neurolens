import os
import cv2
import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras import layers, models

DATA_DIR = os.path.expanduser('~/Downloads/archive (1)')
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def load_data(split):
    X, y = [], []
    for i, emotion in enumerate(EMOTIONS):
        folder = os.path.join(DATA_DIR, split, emotion)
        if not os.path.exists(folder):
            print(f"Missing: {folder}")
            continue
        for fname in os.listdir(folder):
            if not fname.endswith('.jpg') and not fname.endswith('.png'):
                continue
            img = cv2.imread(os.path.join(folder, fname), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (48, 48))
            X.append(img)
            y.append(i)
    return np.array(X), np.array(y)

print("Loading training data...")
X_train, y_train = load_data('train')
print(f"Training samples: {len(X_train)}")

print("Loading test data...")
X_test, y_test = load_data('test')
print(f"Test samples: {len(X_test)}")

# Normalize
X_train = X_train.astype(np.float32) / 255.0
X_test = X_test.astype(np.float32) / 255.0
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

X_train, y_train = shuffle(X_train, y_train, random_state=42)

# Tiny CNN — designed to be fast on ARM
model = models.Sequential([
    layers.Input(shape=(48, 48, 1)),
    layers.Conv2D(16, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(32, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(7, activation='softmax')
])

model.summary()

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Training...")
model.fit(
    X_train, y_train,
    epochs=25,
    batch_size=64,
    validation_data=(X_test, y_test)
)

# Save as TFLite — tiny and fast
print("Converting to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open(os.path.expanduser('~/emotion_tiny.tflite'), 'wb') as f:
    f.write(tflite_model)

print(f"Saved! Size: {len(tflite_model)/1024:.1f} KB")
print("Done! Copy emotion_tiny.tflite to the Pi.")
