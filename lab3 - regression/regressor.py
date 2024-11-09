#The goal of the algorithrm is to predict share price of a chosen company on given day by analyzing trends 
#from 5 days before. 

#Optimization of learning parameters is made using GridSearchCV tool

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor

##############################################################

#Reading input values - ignoring date and name
numpy_x = np.loadtxt('V_data.csv', delimiter=',', skiprows=1, usecols=(1,2,3,5))
numpy_y = np.loadtxt('V_data.csv', delimiter=',', skiprows=6, usecols=(4))

#Normalization
numpy_x = numpy_x/numpy_x.max(axis=0)
numpy_y = numpy_y/numpy_y.max(axis=0)

print("Initial data format is " + str(numpy_x.shape))

#Changind 2D data into 1D array
numpy_x = numpy_x.flatten()
print("Flattening into 1D of " + str(numpy_x.shape) + " elements")

#Making new dataset -  the first elemenet is the first 5 days of the array
numpy_x2 = numpy_x[0:20]

#Adding new elements to the dataset - days 1-6, 2-7 etc.
#Value for days 1-6 is that of 7th day and so on

for x in range (int((len(numpy_x))/4 - 6)):
    numpy_x2 = np.append(numpy_x2, numpy_x[(x+1)*4:(x+1)*4+20])

#Transforming dataset to 2D
numpy_x3 = np.reshape(numpy_x2, (-1, 20))

print("Then resizing into 2D of shape " + str(numpy_x3.shape))

################################################

# k% of all data is testing data
k = 36

#Division into testing and training data (no shuffling - this should probably be improved)
train_X, test_X, train_y, test_y = train_test_split(numpy_x3, numpy_y, shuffle=False, test_size=(k/100))
print("Test data is " + str(len(test_y)*100/len(numpy_y)) + " % of the last data in sheet")
print("Test data length is " + str(len(test_y)) + " days")
print("Train data is " + str(len(train_y)) + " y days and x elements of 5 day sets")

#Setting parameters
parameters = {"learning_rate": ["constant", "adaptive"], 
"activation": ["tanh", "logistic"], 
"random_state": [1], 
"max_iter": [200], 
"solver": ["adam", "sgd", "lbfgs"], 
"learning_rate_init": [0.001, 0.01, 0.1], 
"hidden_layer_sizes" : [(100, ), (80, ), (60, ), (40, ), (20, )] }

#Regression and providing output
regr = GridSearchCV(MLPRegressor(), parameters, n_jobs=-1)

regr.fit(train_X,train_y)
print("Best parameters given by GridSearchCV: ")
print(regr.best_params_)

print("Example results check for 5 first test elements")
print("Estimated values")
print(regr.predict(test_X[:5]))
print("Real values")
print(test_y[:5])

print("Average regression precision is " + str(regr.score(test_X, test_y) * 100) + " %")

#To do - improve results presentation, data shuffling