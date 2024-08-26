#Program uses Adaline model to classify data from Iris dataset, dividing it to 2 classes.
#In the result, the program provides its accuracy as the output

import numpy as np
import pandas as pd

#Class providing Adaline model - suggested by academic supervisor
class AdalineGD(object):
    
    def __init__(self, eta=0.01, epochs=50): 
        self.eta = eta
        self.epochs = epochs

    def train(self, X_train, y):

        self.w_ = np.zeros(1 + X_train.shape[1])
        self.cost_ = []

        for i in range(self.epochs):
            output = self.net_input(X_train)
            errors = (y - output)
            self.w_[1:] += self.eta * X_train.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X_train):
        return np.dot(X_train, self.w_[1:]) + self.w_[0]

    def activation(self, X_train):
        return self.net_input(X_train)

    def predict(self, X_train):
        return np.where(self.activation(X_train) >= 0.0, 1, -1)

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

#Standarization to (0,1)
X_std = np.copy(X_train)
X_std[:,0] = (X_train[:,0] - X_train[:,0].mean()) / X_train[:,0].std()
X_std[:,1] = (X_train[:,1] - X_train[:,1].mean()) / X_train[:,1].std()

X_std2 = np.copy(X_test)
X_std2[:,0] = (X_test[:,0] - X_test[:,0].mean()) / X_test[:,0].std()
X_std2[:,1] = (X_test[:,1] - X_test[:,1].mean()) / X_test[:,1].std()

#Defining neuron - tweaking epochs and eta values changes learning results
ada = AdalineGD(epochs=100, eta=0.01)

#Training
ada.train(X_std, y_train)

print('Incorrect classifications for Adaline Model: %d of %d' % ((y_test != ada.predict(X_std2)).sum(), len(y_test)))

#The results will be different each time, so to test parameters you need to run the code multiple times and
#make some statistics - this process can be of course automated by modifying the script