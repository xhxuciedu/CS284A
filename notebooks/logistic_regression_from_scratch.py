import numpy as np

# 1. Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 2. Cost function for logistic regression
def cost_function(y, y_pred):
    m = len(y)
    return (-1/m) * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

# 3. Gradient descent algorithm
def gradient_descent(X, y, alpha, num_iterations):
    m, n = X.shape
    theta = np.zeros((n, 1))
    cost_history = []

    for _ in range(num_iterations):
        y_pred = sigmoid(np.dot(X, theta))
        gradient = np.dot(X.T, (y_pred - y)) / m
        theta -= alpha * gradient
        cost_history.append(cost_function(y, y_pred))

    return theta, cost_history

# Sample data
np.random.seed(42)
X = np.random.randn(100, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int).reshape(-1, 1)

# Add intercept term
X = np.hstack([np.ones((X.shape[0], 1)), X])

# 4. Train the model
alpha = 0.01
num_iterations = 1000
theta, cost_history = gradient_descent(X, y, alpha, num_iterations)

# Predict
def predict(X, theta):
    return (sigmoid(np.dot(X, theta)) > 0.5).astype(int)

y_pred = predict(X, theta)

accuracy = np.mean(y_pred == y)
print(f"Accuracy: {accuracy * 100:.2f}%")
