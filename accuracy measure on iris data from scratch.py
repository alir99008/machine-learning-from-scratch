

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# normalize the input data
X_train = (X_train - X_train.mean(axis=0)) / X_train.std(axis=0)
X_test = (X_test - X_test.mean(axis=0)) / X_test.std(axis=0)

# convert the labels to one-hot encoding
y_train_one_hot = np.zeros((y_train.size, y_train.max()+1))
y_train_one_hot[np.arange(y_train.size), y_train] = 1

# define the neural network architecture
input_size = X_train.shape[1]
hidden_size = 10
output_size = y_train_one_hot.shape[1]

# initialize the weights and biases randomly
W1 = np.random.randn(input_size, hidden_size)
b1 = np.random.randn(hidden_size)
W2 = np.random.randn(hidden_size, output_size)
b2 = np.random.randn(output_size)

# define the activation function (sigmoid)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# define the derivative of the activation function (sigmoid)
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# set the learning rate and the number of epochs
learning_rate = 0.1
num_epochs = 1000

# train the neural network using backpropagation
for epoch in range(num_epochs):
    # forward propagation
    z1 = np.dot(X_train, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    y_pred = sigmoid(z2)

    # compute the loss and the gradient of the loss with respect to y_pred
    loss = np.square(y_pred - y_train_one_hot).sum()
    dy_pred = 2 * (y_pred - y_train_one_hot)

    # backpropagation
    dW2 = np.dot(a1.T, dy_pred * sigmoid_derivative(z2))
    db2 = (dy_pred * sigmoid_derivative(z2)).sum(axis=0)
    da1 = np.dot(dy_pred * sigmoid_derivative(z2), W2.T)
    dW1 = np.dot(X_train.T, da1 * sigmoid_derivative(z1))
    db1 = (da1 * sigmoid_derivative(z1)).sum(axis=0)

    # update the weights and biases
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    # print the training loss every 100 epochs
    if epoch % 100 == 0:
        print("Epoch {}, Loss: {:.3f}".format(epoch, loss))

# evaluate the trained neural network on the test set
z1 = np.dot(X_test, W1) + b1
a1 = sigmoid(z1)
z2 = np.dot(a1, W2) + b2
y_pred = np.argmax(sigmoid(z2), axis=1)
accuracy = (y_pred == y_test).mean()
print("Test Accuracy: {:.3f}".format(accuracy))
