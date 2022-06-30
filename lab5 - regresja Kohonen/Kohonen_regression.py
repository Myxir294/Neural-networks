#Program wykorzystuje wlasnoręcznie zrealizowany algorytm Kohonena do klasteryzacji danych dotyczących wybranej spółki giełdowej
#Wynikiem działania jest podanie jakosci klasteryzacji (poprzez wynik Daviesa-Bouldina) oraz wizualizacja na wykresach jego
#przełożenia na jakość regresji. Zrealizowano dwa warianty - z wykorzystaniem algorytmu algebraicznego i iteracyjnego

from random import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import davies_bouldin_score
import matplotlib.pyplot as plt

##############################################################

#Algorytm algebraiczny


def Gauss_Function(data,R):
    """Gauss Function of variable R parameter"""
    val = np.exp(-np.square(data/R))
    return val


def Data_Standarization(X):
    """Proper data preprocessing for Kohonen algorithm"""
    sum = 0
    temp = 0
    for each in X:
        sum += each
        temp = temp + 1
    #print(sum)
    avg = sum / len(X)
    X_std = []
    for i in range(len(X)):
        normal = (avg - X[i])
        X_std.append(normal / np.linalg.norm(X[i]))
    return np.array(X_std)

def Kohonen(data, p, alpha_zero, T, norm, alpha_mod, C, C1, C2):
    """Kohonen algorithm for general data"""
    alpha = alpha_zero

    #Inicjalizacja wektorów reprezentantów
    vectors = np.ones((p,data.shape[1]))
    vectors = abs(vectors/(np.sqrt(data.shape[1])))
    chosen_vector = 0

    tmp_tab = [0]*p
    for k in range (T):
        for point in range(len(data)):
            for j in range(p):
                
                #Wybór normy odległości pomiędzy pointem a wektorem
                if(norm == 1):
                    tmp_tab[j] = np.dot(vectors[j],data[point])
                    pom = 0
                if(norm == 2):
                    tmp_tab[j] = np.linalg.norm(vectors[j]-data[point])
                    pom = 1
                if(norm == 3): 
                    sum = 0
                    for k in range(int(data.shape[1])):
                        sum = sum + abs(vectors[j,k]-data[point,k])
                    tmp_tab[j] = np.sqrt(sum)
                    pom = 1

            if(pom != 1):
                value = max(tmp_tab)
            else:
                value = min(tmp_tab)

            chosen_vector = tmp_tab.index(value)

            #Przesuniecie wybranego wektora blizej dopasowanego pointu
            vectors[chosen_vector] = vectors[chosen_vector] + alpha*(data[point] - vectors[chosen_vector])
        
            #Normalizacja
            vectors[chosen_vector] = vectors[chosen_vector]/(np.linalg.norm(vectors[chosen_vector]))

        #Różne zmniejszenia wspolczynnika uczenia
        if(alpha_mod == 1):
            alpha = alpha_zero*(T-(k+1))/T
        if(alpha_mod == 2):
            alpha = alpha_zero*np.exp(-C*(k+1))
        if(alpha_mod == 3):
            alpha = C1/(C2+k+1)

    #Operacje pomocnicze służące do wyznaczenia współczynnika Daviesa-Bouldina -> jakość klasteryzacji
    lengths_temp = [0]*p
    classifications = [0]*len(data)
    for point in range(len(data)):
        for i in range(p):
            lengths_temp[i] = np.linalg.norm(data[point] - vectors[i])
        min_val = min(lengths_temp)
        classifications[point] = lengths_temp.index(min_val)
       
    #Im mniejszy współczynnik tym lepszy podział na klastry
    DB = davies_bouldin_score(data, classifications)
    print("DB score is " + str(DB))
    return vectors


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
k_percent = 25

#Standaryzacja danych wejściowych
numpy_x3 = Data_Standarization(numpy_x3)

#Podzial na zbiory testujace i trenujace - bez przemieszania danych
train_X, test_X, train_y, test_y = train_test_split(numpy_x3, numpy_y, shuffle=False, test_size=(k_percent/100))
print("Test data are " + str(len(test_y)*100/len(numpy_y)) + " % of the last data in sheet")
print("Test data length is " + str(len(test_y)) + " days")
print("Train data are " + str(len(train_y)) + " y days and x elements of 5 day sets")

#Wyznaczenie diam(S) i R - parametru funkcji Gaussa
diam = 0
diam_temp = 0
for i in range(len(train_X)):
    for j in range (len(train_X)):
        diam_temp = np.linalg.norm(train_X[i] - train_X[j])
        if(diam_temp > diam):
            diam = diam_temp

#Podzielenie R przez niewielką liczbę całkowitą
R = diam/1

#Wyznaczenie centrów danych
p = 2
print("Algebraic algorithm")
C = Kohonen(train_X, p, 0.5, 3, 1, 1, 0.5, 0.5, 0.5)

#Wyznaczenie tablicy phi
phi = np.zeros((len(train_X), p), dtype=float)

#Wypelnienie tablicy phi
for i in range(len(phi)):
    for j in range(p):
        phi[i][j] = Gauss_Function((np.linalg.norm(train_X[i] - C[j])),R)

#Wyznaczenie pseudodwrotności
phi_T = np.transpose(phi)

#Wyznaczenie wektora wag

w  = np.dot(np.dot(np.linalg.pinv(np.dot(phi_T,phi)),phi_T),train_y)
#print(w)

#Wyznaczenie wyjscia sieci
network_output = np.zeros(len(test_X))
for i in range(len(test_X)):
    sum = 0
    for j in range(p):
        element = Gauss_Function((np.linalg.norm(test_X[i] - C[j])),R)
        sum = sum + (w[j] * element)
    network_output[i] = sum

#Przedstawienie wyników na wykresie
plt.grid()
plt.plot(test_y, '-', label='Expected Values', c='blue')
plt.plot(network_output, '-', label='Network Output',c= 'red')
plt.title("Number of clusters (algebraic algorithm): " + str(p))
plt.legend()
plt.show()

#####################################################################################

#Algorytm iteracyjny - na podstawie kodu na stronie wskazanej przez prowadzącego

print("Iterative algorithm")

#Funkcja aktywacji sieci RBF
def rbf(x, c, s):
    return np.exp(-1 / (2 * s**2) * (x-c)**2)

#Klasa modelująca sieć RBF wykorzystującą algorytm iteracyjny
class RBFNet(object):

    """Implementation of a Radial Basis Function Network"""
    def __init__(self, k, lr=0.00000001, epochs=200, rbf=rbf):
        self.k = k
        self.lr = lr
        self.epochs = epochs
        self.rbf = rbf
        self.w = np.random.randn(k)
        self.b = np.random.randn(1)

    def fit(self, X, y):
        self.centers = Kohonen(X, self.k, 0.5, 3, 1, 1, 0.5, 0.5, 0.5)
        dMax = max([np.linalg.norm(c1 - c2) for c1 in self.centers for c2 in self.centers])
        self.stds = np.repeat(dMax / np.sqrt(2*self.k), self.k)

    # training
        for epoch in range(self.epochs):
            for i in range(X.shape[0]):
                # forward pass
                a = np.array([self.rbf(X[i], c, s) for c, s, in zip(self.centers, self.stds)])
                F = a.T.dot(self.w) + self.b
                # backward pass
                error = -(y[i] - F)
                # online update
                self.w = self.w - self.lr * a * error
                self.b = self.b - self.lr * error

    #prediction
    def predict(self, X):
        y_pred = []
        for i in range(X.shape[0]):
            a = np.array([self.rbf(X[i], c, s) for c, s, in zip(self.centers, self.stds)])
            F = a.T.dot(self.w) + self.b
            y_pred.append(F)
        return np.array(y_pred)            

#Utworzenie obiektu sieci oraz jej trening (dziala tylko dla k rownego 20, przewiduje sie poprawienie w przyszlosci)
set_epochs = 300
rbfnet = RBFNet(lr=1e-3, k=20, epochs=set_epochs)
rbfnet.fit(train_X, train_y)
y_pred = rbfnet.predict(test_X)

#Wizualizacja danych na wykresie
RBF_Output = y_pred[:,0,0]
plt.plot(test_y, '-', label='Expected Values', c='blue')
plt.plot(RBF_Output, '-', label='RBF-Net output',c='red')
plt.title("Epoch number for RBF network (iterative algorithm): " + str(set_epochs))
plt.legend()
plt.tight_layout()
plt.show()  