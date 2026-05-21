import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# 1. load dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 2. build model
def inception_block(x, filters):
    f1, f3r, f3, f5r, f5, pool_proj = filters

    path1 = layers.Conv2D(f1, (1,1), activation='relu', padding='same')(x)

    path2 = layers.Conv2D(f3r, (1,1), activation='relu', padding='same')(x)
    path2 = layers.Conv2D(f3, (3,3), activation='relu', padding='same')(path2)

    path3 = layers.Conv2D(f5r, (1,1), activation='relu', padding='same')(x)
    path3 = layers.Conv2D(f5, (5,5), activation='relu', padding='same')(path3)

    path4 = layers.MaxPooling2D((3,3), strides=1, padding='same')(x)
    path4 = layers.Conv2D(pool_proj, (1,1), activation='relu', padding='same')(path4)

    return layers.concatenate([path1, path2, path3, path4])

# 3. Build GoogLeNet-like model
inputs = layers.Input(shape=(32,32,3))

x = layers.Conv2D(64,(7,7), strides=2, activation='relu', padding='same')(inputs)
x = layers.MaxPooling2D((3,3), strides=2, padding='same')(x)

x = inception_block(x, [32,32,64,16,32,32])
x = layers.MaxPooling2D((2,2))(x)

x = inception_block(x, [64,48,64,16,32,32])
x = layers.MaxPooling2D((2,2))(x)

x = layers.Flatten()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)

outputs = layers.Dense(10, activation='softmax')(x)

model = models.Model(inputs, outputs)

# 4. Compile
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 5. Train

history = model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=64,
    validation_data=(x_test, y_test)
)

# 6. Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test Accuracy:", test_acc)

# 7. Plot Accuracy / Loss

plt.figure()

plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')

plt.title("GoogLeNet Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')

plt.title("GoogLeNet Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()