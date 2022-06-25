#Program wykorzystuje model Adaline do klasyfikacji danych ze zbioru iris z rozroznieniem na poszczegolne klasy
#Wynikiem działania jest wypisanie ilości niepoprawnych klasyfikacji danych ze zbioru testujacego

import numpy as np
import pandas as pd

#Klasa modelujaca Adaline - wykorzystano propozycje klasy z repozytorium wskazanego przez prowadzącego zajęcia

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

#Wczytanie danych
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
#print(df.columns)


#Zapisanie wyjsc odpowiadajacych wykorzystywanym klasom
#Kazda tablica 150 elementow zawiera 1 w polach odpowiadajacych podanej nazwie i -1 w pozostalych
y2 = np.where(df[4] == 'Iris-setosa', 1, -1)
y3 = np.where(df[4] == 'Iris-versicolor', 1, -1)
y4 = np.where(df[4] == 'Iris-virginica', 1, -1)

#Losowy wybor zestawow danych trenujacych stanowiacych okolo 80% calego zbioru danych
x_train_id = np.random.rand(len(df)) < 0.8
#print(x_train_id)
#print(len(x_train_id))    

#Wyselekcjonowanie pozadanych wartosci dla wybranych danych trenujacych
#Oczekiwana liczebnosc kazdego z ponizszych to 120 
y2_train = y2[x_train_id]
y3_train = y3[x_train_id]
y4_train = y4[x_train_id]

# print(y2_train)
# print(len(y2_train))

#Wyselekcjonowanie pozadanych wartosci dla wybranych danych testujacych
#Negacje zastosowano w celu dopelnienia zbioru danych trenujacych
#Oczekiwana liczebnosc kazdego z ponizszych to 30
y2_test = y2[~x_train_id]
y3_test = y3[~x_train_id]
y4_test = y4[~x_train_id]

# print(y2_test)
# print(len(y2_test))

#Wybor zestawu parametrow decydujacych o klasyfikacji obiektu do jednej z klas
X2_train = df.iloc[x_train_id, [0,1,2,3]].values #Dla danych trenujacych 
X2_test = df.iloc[~x_train_id, [0,1,2,3]].values #Dla danych testujacych


#Standaryzacja danych
X_std = np.copy(X2_train)
X_std[:,0] = (X2_train[:,0] - X2_train[:,0].mean()) / X2_train[:,0].std()
X_std[:,1] = (X2_train[:,1] - X2_train[:,1].mean()) / X2_train[:,1].std()

X_std2 = np.copy(X2_test)
X_std2[:,0] = (X2_test[:,0] - X2_test[:,0].mean()) / X2_test[:,0].std()
X_std2[:,1] = (X2_test[:,1] - X2_test[:,1].mean()) / X2_test[:,1].std()

#Zdefiniowanie neuronów - edycja wartosci epochs i eta bezsposrednio przeklada sie na jakosc nauki
ada1 = AdalineGD(epochs=50, eta=0.001)
ada2 = AdalineGD(epochs=50, eta=0.001)
ada3 = AdalineGD(epochs=50, eta=0.001)

#Trenowanie neuronów
ada1.train(X_std, y2_train)
ada2.train(X_std, y3_train)
ada3.train(X_std, y4_train)

#Wypisanie wynikow dla kazdego z neuronów


print('Liczba blednych klasyfikacji w zbiorze testujacym dla adaline 1: %d' % ((y2_test != ada1.predict(X_std2)).sum()))

#Zapisanie tabeli okresjalacej prawdopodobienstwo nalezenia obiektu dla danej klasy
propability = ada1.net_input(X_std2)

#print(propability)

print('Liczba blednych klasyfikacji w zbiorze testujacym dla adaline 2: %d' % ((y3_test != ada2.predict(X_std2)).sum()))

propability2 = ada2.net_input(X_std2)
#print(propability2)

print('Liczba blednych klasyfikacji w zbiorze testujacym dla adaline 3: %d' % ((y4_test != ada3.predict(X_std2)).sum()))

propability3 = ada3.net_input(X_std2)
#print(propability3)

#Czesc programu odpowiadajaca za zestawienie wynikow
results = []

#Petla dodaje do tablicy results najbardziej prawdopodobne wyniki
for prob1, prob2, prob3 in zip(propability.tolist(), propability2.tolist(), propability3.tolist()): 
  if max(prob1, prob2, prob3) == prob1:
    results.append('Iris-setosa')
  elif max(prob1, prob2, prob3) == prob2:
    results.append('Iris-versicolor')
  elif max(prob1, prob2, prob3) == prob3:
    results.append('Iris-virginica')

#Wizualizacja wynikow - porownanie wartosci uzyskanych z rzeczywistymi
#print(results)

#"Wyciagniecie" z pliku prawdziwych nazw gatunkow 
y_real_value = df.iloc[~x_train_id, 4]

#print(y_real_value)

#Zsumowanie ilosci blednych klasyfikacji
print('Calkowita liczba blednych klasyfikacji uwzgledniajac prawdopodobienstwa: %d na %d' % ((y_real_value != results).sum(), len(results)))
