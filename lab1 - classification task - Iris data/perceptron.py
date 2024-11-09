#Program uses Perceptron model to classify data from Iris dataset, dividing it to 2 classes.
#In the result, the program provides its accuracy as the output

import numpy as np
import pandas as pd

#Class providing Perceptron model - suggested by academic supervisor
class Perceptron(object):
    
    def __init__(self, eta=0.01, epochs=50):
        self.eta = eta
        self.epochs = epochs

    def train(self, X, y):

        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.epochs):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] +=  update * xi
                self.w_[0] +=  update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)

#Reading data
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

#Saving outputs corresponding to each class
#Each table of 150 elements contains 1 if class name is matching, -1 otherwise
y = df.iloc[0:150, 4].values
y = np.where(y == 'Iris-virginica', 1, -1)

#Random choosing of training data - about 80% of all the data, due to standard distribution of [0,1]
x_train_id = np.random.rand(len(df)) < 0.8

#Providing values for training and testing data
y_train = y[x_train_id] # ~ 120 values - training
y_test = y[~x_train_id] # ~ 30 values - testing

#Chossing parameters deciding the classification result
X_train = df.iloc[x_train_id, [0,1,2,3]].values #For training data
X_test = df.iloc[~x_train_id, [0,1,2,3]].values #For testing data

#Defining neuron - tweaking epochs and eta values changes learning results
ppn = Perceptron(epochs=50, eta=0.1)

#Training
ppn.train(X_train, y_train)

print('Total number of misclassifications: %d of %d' % ((y_test != ppn.predict(X_test)).sum(), len(y_test)))

#The results will be different each time, so to test parameters you need to run the code multiple times and
#make some statistics - this process can be of course automated by modifying the script

#With current coefficients, the accuracy varies from ~ 60% to 90%