import numpy as np
from models.Regression import Regression


class LogisticRegression(Regression):
    def __init__(self, theta=np.array([]), learning_rate=0.01, iterations=1000, epochs=50):
        super().__init__(theta)

        self.learning_rate = learning_rate
        self.iterations = iterations
        self.epochs = epochs

    def set_hyperparamters(self, learning_rate=0.01, iterations=1000, epochs=50):
        """
        Change the preexisting model hyperparameters
        :param learning_rate: New learning rate
        :param iterations: New number of training iterations
        :param epochs: New number of training epochs
        """
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.epochs = epochs

    def fit(self, X, y):
        """
        Fit the Logistic Regression using stochastic gradient descent
        :param X: Matrix of shape (n_data + 1, n_features) containing the training data.
        :param y: Matrix of shape (n_features, 1) containing the training labels.
        """
        X = np.c_[np.ones((X.shape[0], 1)), X]  # add x0 = 1 to each instance. This is to account for theta_0

        self.stochastic_gradient_descent(X, y)

    def stochastic_gradient_descent(self, X, y):
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
