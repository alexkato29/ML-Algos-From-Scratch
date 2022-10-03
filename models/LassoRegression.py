"""
Ridge Regression.
"""
import numpy as np
from models.Regression import Regression


class LassoRegression(Regression):
    def __init__(self, theta=np.array([]), method="stochastic", alpha=0.1, learning_rate=0.01, iterations=1000, epochs=50):
        super().__init__(theta)

        self.method = method
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.epochs = epochs

    def set_hyperparamters(self, alpha=0.1, learning_rate=0.01, iterations=1000, epochs=50):
        """
        Change the preexisting model hyperparameters
        :param alpha: The cost of large weights in the model. High values will punish high weights
        :param learning_rate: New learning rate
        :param iterations: New number of training iterations
        :param epochs: New number of training epochs
        """
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.epochs = epochs

    def fit(self, X, y):
        """
        Fit the Lasso Regression
        :param X: Matrix of shape (n_data + 1, n_features) containing the training data.
        :param y: Matrix of shape (n_features, 1) containing the training labels.

        :return: The fitted lasso regression.
        """
        X = np.c_[np.ones((X.shape[0], 1)), X]  # add x0 = 1 to each instance. This is to account for theta_0

        self.fit_stochastic_gradient_descent(X, y)

    def fit_stochastic_gradient_descent(self, X, y):
        m, n = X.shape  # number of instances and features

        if self.theta.shape[0] == 0:  # if theta not yet initialized
            self.theta = np.random.randn(n, 1)  # Returns a random scalar from the Gaussian Dist.

        for epoch in range(self.epochs):
            for i in range(m):
                random_index = np.random.randint(m)  # Massively important that this is a random instance!
                xi = X[random_index:random_index + 1]
                yi = y[random_index:random_index + 1]
                # The difference is, in the gradient, we are adding alpha*sign(theta)
                gradient = 2 * xi.T.dot(xi.dot(self.theta) - yi) + self.alpha * np.sign(self.theta[1:])
                eta = self.learning_schedule(epoch * m + i) / (epoch + 1)
                self.theta = self.theta - eta * gradient
