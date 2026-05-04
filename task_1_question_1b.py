# MNIST Digit Classification
# CCS 2424 - Adaptive Learning Task 1
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

# ── 1. Load & Normalise the MNIST dataset ──────────────────────────
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Flatten 28x28 images to 784-dim vectors and scale to [0, 1]
x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
x_test  = x_test.reshape(-1, 784).astype('float32') / 255.0

# Convert integer labels to one-hot encoded vectors (10 classes)
y_train = keras.utils.to_categorical(y_train, 10)
y_test  = keras.utils.to_categorical(y_test,  10)

# ── 2. Build the model ─────────────────────────────────────────────
model = keras.Sequential([
    # Input layer: 784 features (one per pixel)
    layers.Input(shape=(784,)),
    # Hidden layer 1: 128 neurons, ReLU activation
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),  # Prevent overfitting
    # Hidden layer 2: 64 neurons, ReLU activation
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    # Output layer: 10 neurons (one per digit 0-9), softmax
    layers.Dense(10, activation='softmax')
])

# ── 3. Compile with Adam optimiser & categorical cross-entropy ──────
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()
# ── 4. Train for 10 epochs with 20% validation split ───────────────
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=128,
                    validation_split=0.2,
                    verbose=1)
# ── 5. Evaluate on the test set ────────────────────────────────────
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f'Test accuracy: {test_acc:.4f}')
# ── 6. Visualise training history ──────────────────────────────────
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'],     label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy'); plt.xlabel('Epoch'); plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'],     label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss'); plt.xlabel('Epoch'); plt.legend()
plt.tight_layout()
plt.savefig('mnist_training.png', dpi=150)
plt.show()
