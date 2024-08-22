#Program wykorzystuje wlasnoręcznie zrealizowany algorytm Kohonena do klasteryzacji danych ze zbioru Iris
#Wynikiem działania jest podanie jakosci klasteryzacji (poprzez wynik Daviesa-Bouldina) oraz jego przełożenia na jakość klasyfikacji
#zwizualizowanej poprzez macierz pomyłek

import numpy as np
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.metrics import davies_bouldin_score

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

def Kohonen_Iris(data, target, p, alpha_zero, T, norm, alpha_mod, C, C1, C2):
    """Kohonen algorithm for iris data classification"""
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

    #Wyświetlenie wyników klasyfikacji
    matrix = confusion_matrix(target,classifications)
    print("Iris setosa classification score is " + str(matrix[0][0] * 2) + "%")
    print("Iris virginica classification score is " + str(matrix[1][1] * 2) + "%")
    print("Iris versicolor classification score is " + str(matrix[2][2] * 2) + "%")

    print(matrix)
##############################################################################

#Wczytanie danych
iris = datasets.load_iris()

#Preprocessing
X = Data_Standarization(iris.data)

#Klasteryzacja z klasyfikacją
p = 2
Kohonen_Iris(X, iris.target, p, 0.8, 20, 1, 1, 0.5, 0.5, 0.5)

#uwaga - w następnym zadaniu przedstawiono uniwersalny algorytm Kohonena, bez klasyfikacji 
