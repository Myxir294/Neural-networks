#Program uses multi-layer perceptron to classify data from Iris collection, MNIST and examplar Kaggle.com dataset
#As a result, it provides classification score via confusion matrix
#Best learning parameters are automatically chosen using GridSearchCV tool

#Known issues - data processing method could be tweaked, this may be fixed in the future
#Also the code can produce a lot of the same warnings if iterations number is too small, 
#It could be beneficial to disable it, as it doesn't help much, but I haven't tried it yet
#Program works ok anyway

import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from keras.datasets import mnist
from scipy.special import softmax
from sklearn.metrics import confusion_matrix

iteraions = 20 #default value - tweak it as you wish, less iterations make the code faster but give worse results

#For now, the task is chosen via uncommenting corresponding code section
####################################################

#Data pre-processing
(d3_train_X, train_y), (d3_test_X, test_y) = mnist.load_data()
 
nsamples, nx, ny = d3_train_X.shape
train_X = d3_train_X.reshape((nsamples,nx*ny))
train_X = train_X.astype('float32') / 255

nsamples, nx, ny = d3_test_X.shape
test_X = d3_test_X.reshape((nsamples,nx*ny))
test_X = test_X.astype('float32') / 255

print("For MNIST number images - is the program able to make out written digits?" )

#As the dataset is large, more iterations are unnecessary (like in below line), the results are ok anyway
#Expect the program to work for about a minute or two
iteraions = 5
########################################

# iris = datasets.load_iris()

# train_X, test_X, train_y, test_y = train_test_split(iris.data, iris.target, stratify=iris.target, random_state=5)

# print("For Iris data - is the program able to guess flower species based on measurements data?")
#######################################


# #######################################
# All data except price is used as input
# data = pd.read_csv('train.csv')
# y = data['price_range']
# X = data.drop('price_range', axis='columns')

# train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=5)

# print("For smartphone price tags data (Kaggle.com) - is the program able to guess the phone price?")
# ######################################

#Vector of learing parameters
parameters = {"random_state": [1], "max_iter": [iteraions], "solver": ["adam", "sgd", "lbfgs"], "learning_rate_init": [0.001, 0.01, 0.1], "hidden_layer_sizes" : [(100, ), (80, ), (60, ), (40, ), (20, )] }

#Building classificators with different parameters via GridSearch tool
clf = GridSearchCV(MLPClassifier(), parameters, n_jobs=-1)
clf.out_activation_ = "softmax"  #function used for picking the best classificator possible with given parameters

#Learing
clf.fit(train_X, train_y)

#Results and optimized parameters
print(clf.best_params_)
print("Score is " + str(clf.score(test_X, test_y)*100) + " %")

matrix = confusion_matrix(test_y, clf.predict(test_X))
print(matrix)

for row in matrix:
    print(row / row.sum())

