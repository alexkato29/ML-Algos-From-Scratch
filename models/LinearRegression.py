"""
Standard Linear Regression.
"""
import os
import numpy as np
from datetime import datetime
dir_path = os.path.dirname(os.path.realpath(__file__))


class LinearRegression(object):
    def __init__(self, theta=np.array([]), method="svd", learning_rate=0.01, iterations=1000, epochs=50):
        self.theta = theta
        self.model_name = datetime.now().strftime("%d/%m/%Y-%H:%M")

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

    def learning_schedule(self, t):
        t0, t1 = 5, 50
        return t0 / (t + t1)

    def predict(self, X):
        X = np.c_[np.ones((X.shape[0], 1)), X]  # add x0 = 1 to each instance. This is to account for theta_0
        return X.dot(self.theta)

    def show_model(self):
        print(self.theta)

    def save_model(self):
        f = open(dir_path + "/../saved_models/LinearRegressions.txt", 'w')
        f.write(str(self) + "\n")
        f.close()

    @staticmethod
    def view_stored_models():
        f = open(dir_path + "/../saved_models/LinearRegressions.txt", 'r')
        models = []
        for line in f:
            models.append(line)
        return models

    @staticmethod
    def import_model(model_name):
        f = open(dir_path + "/../saved_models/LinearRegressions.txt", 'r')

        line = f.read()

        while line != model_name:
            line = f.read()

        f.close()
        line = line.split(" ")

        theta = np.array([[float(x)] for x in line[1][1:len(line[1]) - 2].split(",")])

        return LinearRegression(theta=theta)

    def __str__(self):
        to_return = self.model_name + " ["

        for weight in self.theta:
            to_return += str(weight[0]) + ","

        return to_return[:len(to_return) - 2] + "]"

