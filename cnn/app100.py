import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# ==========================================
# 1. Load MNIST dataset
# ==========================================
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# ==========================================
# 2. Normalize data
# ==========================================
X_train = X_train / 255.0
X_test = X_test / 255.0

# ==========================================
# 3. Reshape for CNN
#    (batch, height, width, channels)
# ==========================================
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# ==========================================
# 4. Build CNN model
# ==========================================
model = models.Sequential([

    # Convolution layer
    layers.Conv2D(
        filters=32,
        kernel_size=(3,3),
        activation='relu',
        input_shape=(28,28,1)
    ),

    # Pooling layer
    layers.MaxPooling2D((2,2)),

    # Second convolution
    layers.Conv2D(
        filters=64,
        kernel_size=(3,3),
        activation='relu'
    ),

    # Second pooling
    layers.MaxPooling2D((2,2)),

    # Flatten
    layers.Flatten(),

    # Fully connected layer
    layers.Dense(64, activation='relu'),

    # Output layer
    layers.Dense(10, activation='softmax')
])

# ==========================================
# 5. Compile model
# ==========================================
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ==========================================
# 6. Train model
# ==========================================
history = model.fit(
    X_train,
    y_train,
    epochs=3,
    validation_data=(X_test, y_test)
)

# ==========================================
# 7. Evaluate
# ==========================================
test_loss, test_acc = model.evaluate(X_test, y_test)

print("Test Accuracy:", test_acc)

# ==========================================
# 8. Plot training curve
# ==========================================
plt.figure(figsize=(8,5))

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("CNN Training Accuracy")
plt.legend()
plt.grid()

plt.show()