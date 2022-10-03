import os
import numpy as np
from datetime import datetime
dir_path = os.path.dirname(os.path.realpath(__file__))


class Regression:
    def __init__(self, theta=np.array([])):
        self.theta = theta
        self.model_name = datetime.now().strftime("%d/%m/%Y-%H:%M")

    def predict(self, X):
        X = np.c_[np.ones((X.shape[0], 1)), X]  # add x0 = 1 to each instance. This is to account for theta_0
        return X.dot(self.theta)

    def learning_schedule(self, t):
        t0, t1 = 5, 50
        return t0 / (t + t1)

    def show_model(self):
        print(self)

    def save_model(self):
        f = open(dir_path + "/saved_models/LinearRegressions.txt", 'w')
        f.write(str(self) + "\n")
        f.close()

    @staticmethod
    def view_stored_models():
        f = open(dir_path + "/saved_models/LinearRegressions.txt", 'r')
        models = []
        for line in f:
            models.append(line)
        return models

    @staticmethod
    def import_model(model_name):
        f = open(dir_path + "/saved_models/LinearRegressions.txt", 'r')

        line = f.read()

        while line != model_name:
            line = f.read()

        f.close()
        line = line.split(" ")

        theta = np.array([[float(x)] for x in line[1][1:len(line[1]) - 2].split(",")])

        return Regression(theta=theta)

    def __str__(self):
        to_return = self.model_name + " ["

        for weight in self.theta:
            to_return += str(weight[0]) + ","

        return to_return[:len(to_return) - 2] + "]"
