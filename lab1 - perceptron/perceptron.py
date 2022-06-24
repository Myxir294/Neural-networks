#Program wykorzystuje model perceptronu do klasyfikacji danych ze zbioru iris
#Wynikiem działania jest wypisanie ilości niepoprawnych klasyfikacji danych ze zbioru testujacego

import numpy as np
import pandas as pd

#Klasa modelujaca perceptron - wykorzystano propozycje klasy z repozytorium wskazanego przez prowadzącego zajęcia
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

#Wczytanie danych
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

#Zdefiniowanie perceptronow - edycja wartosci epochs i eta bezsposrednio przeklada sie na jakosc nauki 
ppn = Perceptron(epochs=50, eta=0.1)

#Trenowanie perceptronu
ppn.train(X_train, y_train)

print('Total number of misclassifications: %d of %d' % ((y_test != ppn.predict(X_test)).sum(), len(y_test)))


