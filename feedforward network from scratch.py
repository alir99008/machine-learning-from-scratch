
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)


mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std



def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)




input_size = X_train.shape[1]
print(input_size)
hidden_size = 10
output_size = 3


np.random.seed(42)

W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))

W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))


def forward(X):
    # Input layer to hidden layer
    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)
    print("A1 = ",A1)

    # Hidden layer to output layer
    Z2 = np.dot(A1, W2) + b2
    A2 = softmax(Z2)
    print("A2 = ",A2)
    return A2





n_epochs = 10
batch_size = 32
n_batches = X_train.shape[0] // batch_size
print(n_batches)

for epoch in range(n_epochs):
    for batch in range(n_batches):
        # Get the current batch
        start = batch * batch_size
        end = start + batch_size
        X_batch = X_train[start:end]
        y_batch = y_train[start:end]

        # Forward propagation
        y_pred = forward(X_batch)
        
        

        
       

