#Program wykorzystuje model neuronu AdalineGD do klasyfikacji danych ze zbioru iris
#Wynikiem działania jest wypisanie ilości niepoprawnych klasyfikacji danych ze zbioru testujacego

import numpy as np
import pandas as pd

#Klasa modelujaca Adaline - wykorzystano propozycje klasy z repozytorium wskazanego przez prowadzącego zajęcia
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

#Zapisanie wyjsc odpowiadajacych wykorzystywanej klasie
#Ponizsza tablica 150 elementow zawiera 1 w polach odpowiadajacych podanej nazwie i -1 w pozostalych
y = df.iloc[0:150, 4].values
y = np.where(y == 'Iris-virginica', 1, -1)

#Losowy wybor zestawow danych trenujacych stanowiacych okolo 80% calego zbioru danych
x_train_id = np.random.rand(len(df)) < 0.8

#Wyselekcjonowanie pozadanych wartosci dla wybranych danych trenujacych i testujacych
y_train = y[x_train_id] #dlugosc ok 120
y_test = y[~x_train_id] #dlugosc ok 30

#Wybor zestawu parametrow decydujacych o klasyfikacji obiektu do jednej z klas
X_train = df.iloc[x_train_id, [0,1,2,3]].values #Dla danych trenujacych 
X_test = df.iloc[~x_train_id, [0,1,2,3]].values #Dla danych testujacych

#Standaryzacja danych
X_std = np.copy(X_train)
X_std[:,0] = (X_train[:,0] - X_train[:,0].mean()) / X_train[:,0].std()
X_std[:,1] = (X_train[:,1] - X_train[:,1].mean()) / X_train[:,1].std()

X_std2 = np.copy(X_test)
X_std2[:,0] = (X_test[:,0] - X_test[:,0].mean()) / X_test[:,0].std()
X_std2[:,1] = (X_test[:,1] - X_test[:,1].mean()) / X_test[:,1].std()

#Zdefiniowanie modelu Adeline - edycja wartosci epochs i eta bezsposrednio przeklada sie na jakosc nauki 
ada = AdalineGD(epochs=100, eta=0.01)

ada.train(X_std, y_train)

print('Incorrect classifications for Adaline 2: %d of %d' % ((y_test != ada.predict(X_std2)).sum(), len(y_test)))

