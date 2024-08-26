#Program uses Adaline model to classify data from Iris dataset, dividing it to 3 separate classes.
#In the result, the program provides its accuracy as the output

import numpy as np
import pandas as pd

#Class providing Adaline model - suggested by academic supervisor

class AdalineGD(object):
    
    def __init__(self, eta=0.01, epochs=50): 
        self.eta = eta
        self.epochs = epochs

    def train(self, X2_train, y):

        self.w_ = np.zeros(1 + X2_train.shape[1])
        self.cost_ = []

        for i in range(self.epochs):
            output = self.net_input(X2_train)
            errors = (y - output)
            self.w_[1:] += self.eta * X2_train.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X2_train):
        return np.dot(X2_train, self.w_[1:]) + self.w_[0]

    def activation(self, X2_train):
        return self.net_input(X2_train)

    def predict(self, X2_train):
        return np.where(self.activation(X2_train) >= 0.0, 1, -1)

#Reading data
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
#print(df.columns)

#Saving outputs corresponding to each class
#Each table of 150 elements contains 1 if class name is matching, -1 otherwise

y2 = np.where(df[4] == 'Iris-setosa', 1, -1)
y3 = np.where(df[4] == 'Iris-versicolor', 1, -1)
y4 = np.where(df[4] == 'Iris-virginica', 1, -1)

#Random choosing of training data - about 80% of all the data, due to standard distribution of [0,1]
x_train_id = np.random.rand(len(df)) < 0.8
#print(x_train_id)
#print(len(x_train_id))    

#Providing values for training data - each vector is about 120 elements long

y2_train = y2[x_train_id]
y3_train = y3[x_train_id]
y4_train = y4[x_train_id]

# print(y2_train)
# print(len(y2_train))

#Providing values for testing data
#Each vector is about 30 elements long

y2_test = y2[~x_train_id]
y3_test = y3[~x_train_id]
y4_test = y4[~x_train_id]

# print(y2_test)
# print(len(y2_test))

#Chossing parameters deciding the classification result
X2_train = df.iloc[x_train_id, [0,1,2,3]].values #For training data
X2_test = df.iloc[~x_train_id, [0,1,2,3]].values #For testing data

#Standarization to (0,1)
X_std = np.copy(X2_train)
X_std[:,0] = (X2_train[:,0] - X2_train[:,0].mean()) / X2_train[:,0].std()
X_std[:,1] = (X2_train[:,1] - X2_train[:,1].mean()) / X2_train[:,1].std()

X_std2 = np.copy(X2_test)
X_std2[:,0] = (X2_test[:,0] - X2_test[:,0].mean()) / X2_test[:,0].std()
X_std2[:,1] = (X2_test[:,1] - X2_test[:,1].mean()) / X2_test[:,1].std()

#Defining neurons - tweaking epochs and eta values changes learning results
ada1 = AdalineGD(epochs=50, eta=0.001)
ada2 = AdalineGD(epochs=50, eta=0.001)
ada3 = AdalineGD(epochs=50, eta=0.001)

#Training
ada1.train(X_std, y2_train)
ada2.train(X_std, y3_train)
ada3.train(X_std, y4_train)

#Output results

print('Incorrect test data classifications for Adaline 1: %d' % ((y2_test != ada1.predict(X_std2)).sum()))

#Saving tables of probability
probability = ada1.net_input(X_std2)

#print(probability)

print('Incorrect test data classifications for Adaline 2: %d' % ((y3_test != ada2.predict(X_std2)).sum()))

probability2 = ada2.net_input(X_std2)
#print(probability2)

print('Incorrect test data classifications for Adaline 3: %d' % ((y4_test != ada3.predict(X_std2)).sum()))

probability3 = ada3.net_input(X_std2)
#print(probability3)

#Output data processing 
results = []

#This loop adds most probable results to the 'results' list
for prob1, prob2, prob3 in zip(probability.tolist(), probability2.tolist(), probability3.tolist()): 
  if max(prob1, prob2, prob3) == prob1:
    results.append('Iris-setosa')
  elif max(prob1, prob2, prob3) == prob2:
    results.append('Iris-versicolor')
  elif max(prob1, prob2, prob3) == prob3:
    results.append('Iris-virginica')

#Data presentation - comparison of output values and real ones

#print(results)

#"Extraction" of real species names from the file 
y_real_value = df.iloc[~x_train_id, 4]

#print(y_real_value)

#Counting of errors - how efficient we are
print('Total number of incorrect classifications (by probability) : %d of %d' % ((y_real_value != results).sum(), len(results)))

