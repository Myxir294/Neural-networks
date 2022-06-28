#Program wykorzystuje perceptron wielowarstwowy do wykonania regresji danych dotyczących cen akcji dla wybranej spółki
#Celem algorytmu jest umozliwienie przewidywanie jednej z cen w danym dniu na podstawie znajomosci danych z 5 poprzednich dni
#Wynikiem działania jest podanie dobranych parametrów oraz dokładności uzyskanego wyniku
#Najlepsze parametry algorytmu dobierane są automatycznie z wykorzystaniem narzędzia GridSearchCV

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor

##############################################################

#Wczytanie danych wejsciowych - ignorowanie daty oraz nazwy
numpy_x = np.loadtxt('V_data.csv', delimiter=',', skiprows=1, usecols=(1,2,3,5))
numpy_y = np.loadtxt('V_data.csv', delimiter=',', skiprows=6, usecols=(4))

#Normalizacja danych
numpy_x = numpy_x/numpy_x.max(axis=0)
numpy_y = numpy_y/numpy_y.max(axis=0)

print("Initial data format is " + str(numpy_x.shape))

#Zamiana danych 2D w szereg 1D
numpy_x = numpy_x.flatten()
print("Flattening into 1D of " + str(numpy_x.shape) + " elements")

#Stworzenie nowego zestawu danych ktorego pierwszy element to pierwsze piec dni szeregu
numpy_x2 = numpy_x[0:20]

#Dodawanie kolejnych elementow do zestawu danych - dni 1-6, 2-7 itd.
#Dniom 0-5 odpowiada wartosc ceny zamkniecia z dnia 6, dniom 1-6 z dnia 7 itd.
for x in range (int((len(numpy_x))/4 - 6)):
    numpy_x2 = np.append(numpy_x2, numpy_x[(x+1)*4:(x+1)*4+20])

#Transformacja zestawu danych do formy 2D
numpy_x3 = np.reshape(numpy_x2, (-1, 20))

print("Then resizing into 2D of shape " + str(numpy_x3.shape))

################################################

# k% wszystkich danych to dane testujace
k = 36

#Podzial na zbiory testujace i trenujace - bez przemieszania danych
train_X, test_X, train_y, test_y = train_test_split(numpy_x3, numpy_y, shuffle=False, test_size=(k/100))
print("Test data are " + str(len(test_y)*100/len(numpy_y)) + " % of the last data in sheet")
print("Test data length is " + str(len(test_y)) + " days")
print("Train data are " + str(len(train_y)) + " y days and x elements of 5 day sets")

#Wektor parametrow
parameters = {"learning_rate": ["constant", "adaptive"], 
"activation": ["tanh", "logistic"], 
"random_state": [1], 
"max_iter": [200], 
"solver": ["adam", "sgd", "lbfgs"], 
"learning_rate_init": [0.001, 0.01, 0.1], 
"hidden_layer_sizes" : [(100, ), (80, ), (60, ), (40, ), (20, )] }


#Regresja i podanie wynikow
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

#Przewiduje sie możliwość rozszerzenia programu o ponowne przeskalowanie danych,
#tak aby uzyskane wyniki można było łatwiej odnieść do rzeczywistości.
