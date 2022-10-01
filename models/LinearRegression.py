"""
Standard Linear Regression.
"""
import numpy as np


class LinearRegression(object):
    def __init__(self, method="svd", learning_rate=0.01, iterations=1000):
        self.theta = np.array([])
        self.method = method
        self.learning_rate = learning_rate
        self.iterations = iterations

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
        elif self.method == "gradient descent":
            self.fit_gradient_descent(X, y)

    # See README for derivation of the equation
    def fit_normal(self, X, y):
        self.theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

    # See singular value decomposition
    def fit_svd(self, X, y):
        self.theta = np.linalg.pinv(X).dot(y)

    def fit_gradient_descent(self, X, y):
        m, n = X.shape  # number of instances and features

        if self.theta.shape[0] == 0:  # if theta not yet initialized
            self.theta = np.random.randn(n, 1)  # Returns a random scalar from the Gaussian Dist. (mu, sigma)

        for i in range(self.iterations):
            gradients = (2/m) * X.T.dot(X.dot(self.theta) - y)
            self.theta = self.theta - self.learning_rate * gradients

    def predict(self, X):
        X = np.c_[np.ones((X.shape[0], 1)), X]  # add x0 = 1 to each instance. This is to account for theta_0
        return X.dot(self.theta)

    def show_model(self):
        print(self.theta)

    # TODO: save the model to a file for later use without retraining
    def save_model(self, model_name):
        pass

    # TODO: view all models available for import
    def view_stored_models(self):
        pass

    # TODO: import a model from a save file to avoid retraining
    def import_model(self, model_name):
        pass
