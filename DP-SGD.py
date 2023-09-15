import copy

import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, ReLU, Softmax
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.utils import to_categorical
import math
import copy

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Preprocess the data
X_train = X_train.reshape(X_train.shape[0], -1).astype('float32') / 255.0
X_test = X_test.reshape(X_test.shape[0], -1).astype('float32') / 255.0
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=60)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# Build the model
model = Sequential()
model.add(Dense(1000, activation='relu', input_shape=(60,)))
model.add(Dense(10, activation='softmax'))

# Define the loss function
loss_fn = CategoricalCrossentropy()

# Define the optimizer
optimizer = tf.keras.optimizers.SGD(learning_rate=0.05)

# Compile the model
model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

# Training loop

batch_size = 300
num_batches = len(X_train) // batch_size  # m
epochs = 50

epsilon_list = np.random.uniform(low=2, high=2, size=(num_batches, 1))  # Local privacy budget epsilon0
accuracy_vector = np.zeros([epochs, 1])
for epoch in range(epochs):
    epoch_loss = 0.0
    epoch_accuracy = 0.0
    for batch in range(num_batches):
        epsilon0 = epsilon_list[batch]
        delta0 = 10 ** (-5)
        C = 10
        sigma = 2 * C / batch_size * math.sqrt(2 * math.log(1.25 / delta0)) / epsilon0 #gaussian noise
        sigma1 = 2 * C / batch_size / epsilon0  #Laplace noise
        start = batch * batch_size
        end = start + batch_size
        x_batch = X_train[start:end]
        y_batch = y_train[start:end]

        with tf.GradientTape() as tape:
            # Forward pass
            logits = model(x_batch, training=True)
            # Compute the loss value
            loss_value = loss_fn(y_batch, logits)

        # Compute gradients
        gradients = tape.gradient(loss_value, model.trainable_variables)
        # Calculate gradient norm
        # gradient_norm = tf.linalg.global_norm(gradients) #2-norm
        gradient_norm = 0
        for tf_i in range(4):
            temp_gradient_norm = np.linalg.norm(gradients[tf_i], ord=1)  # 1-norm
            gradient_norm += temp_gradient_norm
        # Clip gradients if necessary
        if gradient_norm > C:
            gradients = [gradient * (C / gradient_norm) for gradient in gradients]

        # Add Gaussian noise to gradients
        noisy_gradients = copy.copy(gradients)
        noisy_gradients[0] = gradients[0] + np.random.laplace(0, sigma1, gradients[0].shape)
        noisy_gradients[1] = gradients[1] + np.random.laplace(0, sigma1, gradients[1].shape)
        noisy_gradients[2] = gradients[2] + np.random.laplace(0, sigma1, gradients[2].shape)
        noisy_gradients[3] = gradients[3] + np.random.laplace(0, sigma1, gradients[3].shape)
        # noisy_gradients = [gradient + tf.random.normal(gradient.shape, mean=0.0, stddev=sigma) for gradient in
        # gradients]
        # Apply gradients
        optimizer.apply_gradients(zip(noisy_gradients, model.trainable_variables))

        # Update metrics
        epoch_loss += loss_value.numpy()
        epoch_accuracy += np.mean(np.argmax(logits, axis=1) == np.argmax(y_batch, axis=1))

    epoch_loss /= num_batches
    epoch_accuracy /= num_batches

    print('Epoch {}/{} - Loss: {:.4f} - Accuracy: {:.4f}'.format(epoch + 1, epochs, epoch_loss, epoch_accuracy))
    accuracy_vector[epoch] = epoch_accuracy

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(batch_size)
print('Test Loss: {:.4f} - Test Accuracy: {:.4f}'.format(test_loss, test_accuracy))
# print(accuracy_vector)


