import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt



# 1. Load dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 2. print shape

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

# 3. Show ONE image

plt.figure()
plt.imshow(X_train[0],cmap='gray')
plt.title(f"Label:{y_train[0]}")
plt.axis('off')
plt.show()

# 4. Show MULTIPLE images (grid)

for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(X_train[i],cmap='gray')
    plt.title(str(y_train[i]))
    plt.axis('off')

plt.tight_layout()
plt.show()