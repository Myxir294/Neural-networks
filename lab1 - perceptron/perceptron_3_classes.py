#Program uses Perceptron model to classify data from Iris dataset, dividing it to 3 separate classes.
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
X2_train = df.iloc[x_train_id, [0,1,2,3]].values #Dla danych trenujacych 
X2_test = df.iloc[~x_train_id, [0,1,2,3]].values #Dla danych testujacych

#Defining neurons - tweaking epochs and eta values changes learning results
ppn2 = Perceptron(epochs=50, eta=0.001)
ppn3 = Perceptron(epochs=50, eta=0.001)
ppn4 = Perceptron(epochs=50, eta=0.001)

#Training
ppn2.train(X2_train, y2_train)
ppn3.train(X2_train, y3_train)
ppn4.train(X2_train, y4_train)

#Output results

# print('Weights: %s' % ppn2.w_)
# plot_decision_regions(X2_test, y2_test, clf=ppn2)
# plt.title('Perceptron 1 - Iris-setosa classification')
# plt.xlabel('sepal length [cm]')
# plt.ylabel('petal length [cm]')
# plt.show()

print('Incorrect test data classifications for perceptron 1: %d' % ((y2_test != ppn2.predict(X2_test)).sum()))

#Saving tables of probability
propability = ppn2.net_input(X2_test)

#print(propability)

print('Incorrect test data classifications for perceptron 2: %d' % ((y3_test != ppn3.predict(X2_test)).sum()))

propability2 = ppn3.net_input(X2_test)
#print(propability2)

print('Incorrect test data classifications for perceptron 3: %d' % ((y4_test != ppn4.predict(X2_test)).sum()))

propability3 = ppn4.net_input(X2_test)
#print(propability3)

#Output data processing 
results = []

#This loop adds most probable results to the 'results' list
for prob1, prob2, prob3 in zip(propability.tolist(), propability2.tolist(), propability3.tolist()): 
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
print('Total number of misclassifications (by probability) : %d of %d' % ((y_real_value != results).sum(), len(results)))
