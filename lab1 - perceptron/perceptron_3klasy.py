#Program wykorzystuje model perceptronu do klasyfikacji danych ze zbioru iris z rozroznieniem na poszczegolne klasy
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

#Zdefiniowanie perceptronow - edycja wartosci epochs i eta bezsposrednio przeklada sie na jakosc nauki
ppn2 = Perceptron(epochs=50, eta=0.001)
ppn3 = Perceptron(epochs=50, eta=0.001)
ppn4 = Perceptron(epochs=50, eta=0.001)

#Trenowanie perceptronow
ppn2.train(X2_train, y2_train)
ppn3.train(X2_train, y3_train)
ppn4.train(X2_train, y4_train)

#Wypisanie wynikow dla kazdego z perceptronow

# print('Weights: %s' % ppn2.w_)
# plot_decision_regions(X2_test, y2_test, clf=ppn2)
# plt.title('Perceptron 1 - Klasyfikacja do Iris-setosa')
# plt.xlabel('sepal length [cm]')
# plt.ylabel('petal length [cm]')
# plt.show()

print('Incorrect test data classifications for perceptron 1: %d' % ((y2_test != ppn2.predict(X2_test)).sum()))

#Zapisanie tabeli okresjalacej prawdopodobienstwo nalezenia obiektu dla danej klasy
propability = ppn2.net_input(X2_test)

#print(propability)

print('Incorrect test data classifications for perceptron 2: %d' % ((y3_test != ppn3.predict(X2_test)).sum()))

propability2 = ppn3.net_input(X2_test)
#print(propability2)

print('Incorrect test data classifications for perceptron 3: %d' % ((y4_test != ppn4.predict(X2_test)).sum()))

propability3 = ppn4.net_input(X2_test)
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
print('Total number of misclassifications (by probability) : %d na %d' % ((y_real_value != results).sum(), len(results)))
