from tensorflow.keras.datasets import mnist # type: ignore
import matplotlib.pyplot as plt # type: ignore

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Show the first image in training set
plt.imshow(x_train[0], cmap='gray')
plt.title(f"Label: {y_train[0]}")
plt.show()
