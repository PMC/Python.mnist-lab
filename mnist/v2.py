import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

# 1. Load the data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. Normalize the images
x_train = x_train / 255.0
x_test = x_test / 255.0

# 3. One-hot encode the labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 4. Build the model
model = Sequential(
    [
        Flatten(input_shape=(28, 28)),  # Flatten 28x28 to 784
        Dense(128, activation="relu"),  # Hidden layer
        Dense(10, activation="softmax"),  # Output layer for 10 classes
    ]
)

# 5. Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# 6. Train the model
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))

# 7. Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")
