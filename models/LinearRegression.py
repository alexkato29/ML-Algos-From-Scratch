"""
Standard Linear Regression.
"""
import numpy as np
from models.Regression import Regression


class LinearRegression(Regression):
    def __init__(self, theta=np.array([]), method="svd", learning_rate=0.01, iterations=1000, epochs=50):
        super().__init__(theta)

        self.method = method
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.epochs = epochs

    def set_hyperparamters(self, method="svd", learning_rate=0.01, iterations=1000, epochs=50):
        """
        Change the preexisting model hyperparameters
        :param method: New method of training. Acceptable values: svd, normal, gradient, stochastic, mini-batch
        :param learning_rate: New learning rate
        :param iterations: New number of training iterations
        :param epochs: New number of training epochs
        """
        self.method = method
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.epochs = epochs

    def fit(self, X, y):
        """
        Fit the Linear Regression
        :param X: Matrix of shape (n_data + 1, n_features) containing the training data.
        :param y: Matrix of shape (n_features, 1) containing the training labels.

        :return: The fitted linear regression.
        """
        X = np.c_[np.ones((X.shape[0], 1)), X]  # add x0 = 1 to each instance. This is to account for theta_0

        if self.method == "svd":
            self.fit_svd(X, y)
        elif self.method == "normal":
            self.fit_normal(X, y)
        elif self.method == "gradient":
            self.fit_gradient_descent(X, y)
        elif self.method == "stochastic":
            self.fit_stochastic_gradient_descent(X, y)
        elif self.method == "mini-batch":
            self.fit_mini_batch_gradient_descent(X, y)

    # See README for derivation of the equation
    def fit_normal(self, X, y):
        self.theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

    # TODO: Add svd derivation
    def fit_svd(self, X, y):
        self.theta = np.linalg.pinv(X).dot(y)

    # See gradient descent derivation
    def fit_gradient_descent(self, X, y):
        m, n = X.shape  # number of instances and features

        if self.theta.shape[0] == 0:  # if theta not yet initialized
            self.theta = np.random.randn(n, 1)  # Returns a random scalar from the Gaussian Dist.

        for i in range(self.iterations):
            gradients = (2 / m) * X.T.dot(X.dot(self.theta) - y)
            self.theta = self.theta - self.learning_rate * gradients

    # Same as gradient descent, except use only one instance in the data rather than all the data
    def fit_stochastic_gradient_descent(self, X, y):
        m, n = X.shape  # number of instances and features

        if self.theta.shape[0] == 0:  # if theta not yet initialized
            self.theta = np.random.randn(n, 1)  # Returns a random scalar from the Gaussian Dist.

        for epoch in range(self.epochs):
            for i in range(m):
                random_index = np.random.randint(m)  # Massively important that this is a random instance!
                xi = X[random_index:random_index + 1]
                yi = y[random_index:random_index + 1]
                gradient = 2 * xi.T.dot(xi.dot(self.theta) - yi)  # NOTE: it is 2 NOT 2/m
                eta = self.learning_schedule(epoch * m + i)
                self.theta = self.theta - eta * gradient

    def fit_mini_batch_gradient_descent(self, X, y):
        m, n = X.shape  # number of instances and features
        batch_size = 32

        if self.theta.shape[0] == 0:  # if theta not yet initialized
            self.theta = np.random.randn(n, 1)  # Returns a random scalar from the Gaussian Dist.

        for epoch in range(self.epochs):
            for i in range(m):
                random_indices = np.random.choice(m, size=batch_size, replace=False)
                X_i = X[random_indices, :]
                yi = y[random_indices, :]
                gradient = (2/batch_size) * X_i.T.dot(X_i.dot(self.theta) - yi)  # NOTE: it is 2/batch_size
                eta = self.learning_schedule(epoch * m + i)
                self.theta = self.theta - eta * gradient

