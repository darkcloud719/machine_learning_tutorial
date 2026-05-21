import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# =========================
# 1. Load dataset
# =========================
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# =========================
# 2. Residual Block
# =========================
def res_block(x, filters, downsample=False):
    shortcut = x

    strides = 2 if downsample else 1

    # First conv
    x = layers.Conv2D(filters, (3,3), strides=strides, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)

    # Second conv
    x = layers.Conv2D(filters, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)

    # Match dimensions if needed
    if downsample or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1,1), strides=strides, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    # Skip connection
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)

    return x

# =========================
# 3. Build ResNet model
# =========================
inputs = layers.Input(shape=(32,32,3))

x = layers.Conv2D(64, (3,3), padding='same', activation='relu')(inputs)
x = layers.BatchNormalization()(x)

# Residual blocks
x = res_block(x, 64)
x = res_block(x, 64)

x = res_block(x, 128, downsample=True)
x = res_block(x, 128)

x = res_block(x, 256, downsample=True)
x = res_block(x, 256)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x)

outputs = layers.Dense(10, activation='softmax')(x)

model = models.Model(inputs, outputs)

# =========================
# 4. Compile
# =========================
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# =========================
# 5. Train
# =========================
history = model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=64,
    validation_data=(x_test, y_test)
)

# =========================
# 6. Evaluate
# =========================
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test accuracy:", test_acc)

# =========================
# 7. 📊 Plot Accuracy
# =========================
plt.figure()
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title("ResNet Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# =========================
# 8. 📉 Plot Loss
# =========================
plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("ResNet Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()