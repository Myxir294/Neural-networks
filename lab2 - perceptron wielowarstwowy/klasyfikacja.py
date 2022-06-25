#Program wykorzystuje perceptron wielowarstwowy do klasyfikacji danych ze zbiorów Iris, MNIST oraz wybranego z serwisu Kaggle.com
#Wynikiem działania jest zwizualizowanie poprawności klasyfikacji poprzez macierze pomyłek
#Najlepsze parametry algorytmu dobierane są automatycznie z wykorzystaniem narzędzia GridSearchCV

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from keras.datasets import mnist
from scipy.special import softmax
from sklearn.metrics import confusion_matrix


#Wybór zadania odbywa się poprzez odkomentowanie właściwej sekcji wczytania danych

####################################################

# (d3_train_X, train_y), (d3_test_X, test_y) = mnist.load_data()
 
# nsamples, nx, ny = d3_train_X.shape
# train_X = d3_train_X.reshape((nsamples,nx*ny))
# train_X = train_X.astype('float32') / 255

# nsamples, nx, ny = d3_test_X.shape
# test_X = d3_test_X.reshape((nsamples,nx*ny))
# test_X = test_X.astype('float32') / 255

# print("Dla obrazków ze zbioru MNIST" )

# Ważne - ustawić liczbę iteracji "max_iter" na niewielką, np. 5 - w innym przypadku niepotrzebnie długo liczy
########################################

iris = datasets.load_iris()

train_X, test_X, train_y, test_y = train_test_split(iris.data, iris.target, stratify=iris.target, random_state=5)

print("Dla danych Iris")
#######################################


# #######################################
# data = pd.read_csv('train.csv')
# y = data['price_range']
# X = data.drop('price_range', axis='columns')

# train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=5)

# print("Dla danych dotyczących przedziału ceny telefonu")
# ######################################

#Wektor przyjętych parametrów
parameters = {"random_state": [1], "max_iter": [50], "solver": ["adam", "sgd", "lbfgs"], "learning_rate_init": [0.001, 0.01, 0.1], "hidden_layer_sizes" : [(100, ), (80, ), (60, ), (40, ), (20, )] }

#Zbudowanie klasyfikatora
clf = GridSearchCV(MLPClassifier(), parameters, n_jobs=-1)
clf.out_activation_ = "softmax" 

#Nauka
clf.fit(train_X, train_y)

#Wyświetlenie wyników i dobranych parametrów
print(clf.best_params_)
print("Dokladnosc to " + str(clf.score(test_X, test_y)*100) + " %")

matrix = confusion_matrix(test_y, clf.predict(test_X))
print(matrix)

for row in matrix:
    print(row / row.sum())