import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# 1. Load dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 2. Build VGG-like model
model = models.Sequential()

# Block 1
model.add(layers.Conv2D(64, (3,3), activation='relu', padding='same', input_shape=(32,32,3)))
model.add(layers.Conv2D(64, (3,3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2,2)))

# Block 2
model.add(layers.Conv2D(128, (3,3), activation='relu', padding='same'))
model.add(layers.Conv2D(128, (3,3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2,2)))

# Block 3
model.add(layers.Conv2D(256, (3,3), activation='relu', padding='same'))
model.add(layers.Conv2D(256, (3,3), activation='relu', padding='same'))
model.add(layers.Conv2D(256, (3,3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2,2)))

# Block 4
model.add(layers.Conv2D(512, (3,3), activation='relu', padding='same'))
model.add(layers.Conv2D(512, (3,3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2,2)))

# 3. Fully Connected layers
model.add(layers.Flatten())
model.add(layers.Dense(4096, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(4096, activation='relu'))
model.add(layers.Dropout(0.5))

# Output
model.add(layers.Dense(10, activation='softmax'))

# 4. Compile
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 5. Train
history = model.fit(
    x_train, y_train,
    epochs=1,
    batch_size=64,
    validation_data=(x_test, y_test)
)

# 6. Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test)

print("Test Accuracy:", test_acc)

# 7. Plot Accuracy / Loss
plt.figure()

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_loss'], label='Validation Loss')

plt.title('VGG Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()