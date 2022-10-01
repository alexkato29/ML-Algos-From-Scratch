"""
Standard Linear Regression.
"""
import numpy as np


class LinearRegression(object):
    def __init__(self):
        self.theta = np.array([])

    def fit(self, X, y, method="normal"):
        """
        Fit the Linear Regression
        :param X: Matrix of shape (n_data + 1, n_features) containing the training data.
        :param y: Matrix of shape (n_features, 1) containing the training labels.
        :param method: The method by which to fit the regression. Defaults to the normal equation.
        :return: The fitted linear regression.
        """
        if method == "normal":
            self.fit_normal(X, y)

    # See README for derivation of the equation
    def fit_normal(self, X, y):
        self.theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

    def predict(self, X):
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