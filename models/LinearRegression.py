"""
Standard Linear Regression.
"""
import os
import numpy as np
from datetime import datetime
dir_path = os.path.dirname(os.path.realpath(__file__))


class LinearRegression(object):
    def __init__(self, theta=np.array([])):
        self.theta = theta
        self.model_name = datetime.now().strftime("%d/%m/%Y-%H:%M")

    def fit(self, X, y, method="svd", learning_rate=0.01, iterations=1000):
        """
        Fit the Linear Regression
        :param X: Matrix of shape (n_data + 1, n_features) containing the training data.
        :param y: Matrix of shape (n_features, 1) containing the training labels.
        :param method: String defining what training method should be used.
        :param learning_rate: Learning rate of the model (when applicable).
        :param iterations: Number of iterations of learning (when applicable).

        :return: The fitted linear regression.
        """
        X = np.c_[np.ones((X.shape[0], 1)), X]  # add x0 = 1 to each instance. This is to account for theta_0

        if method == "svd":
            self.fit_svd(X, y)
        elif method == "normal":
            self.fit_normal(X, y)
        elif method == "gradient descent":
            self.fit_gradient_descent(X, y, learning_rate, iterations)

    # See README for derivation of the equation
    def fit_normal(self, X, y):
        self.theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

    # See singular value decomposition
    def fit_svd(self, X, y):
        self.theta = np.linalg.pinv(X).dot(y)

    # See gradient descent derivation
    def fit_gradient_descent(self, X, y, learning_rate, iterations):
        m, n = X.shape  # number of instances and features

        if self.theta.shape[0] == 0:  # if theta not yet initialized
            self.theta = np.random.randn(n, 1)  # Returns a random scalar from the Gaussian Dist. (mu, sigma)

        for i in range(iterations):
            gradients = (2 / m) * X.T.dot(X.dot(self.theta) - y)
            self.theta = self.theta - learning_rate * gradients

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

