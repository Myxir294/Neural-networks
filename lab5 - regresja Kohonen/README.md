# Kohonen-regression
Folder zawiera program realizujący alternatywne podejście do zadania 3., dotyczącego regresji dla danych wybranej spółki giełdowej, dostępnych pod adresem https://www.kaggle.com/datasets/camnugent/sandp500

Regresja odbywa się z wykorzystaniem napisanego w poprzednim zadaniu algorytmu Kohonena oraz typu sieci Radial Basis Function  

W zadaniu zrealizowano dwa warianty - wykorzystujący ręcznie napisany algorytm algebraiczny na podstawie materiałów pomocniczych do kursu oraz program oparty na algorytmie iteracyjnym ze strony https://pythonmachinelearning.pro/using-neural-networks-for-regression-radial-basis-function-networks/  

Wynikiem działania jest podanie jakości klasteryzacji (poprzez wynik Daviesa-Bouldina) oraz wizualizacja na wykresach jego
przełożenia na jakość regresji.   

UWAGA - z powodu trudności związanych z przerobieniem wskazanego kodu tak, aby mógł obsługiwać dane wielowymiarowe, realizacja algorytmu iteracyjnego działa dla ustalonej liczby klastrów wynoszącej 20. Zbieżność wyjść skonstruowanej sieci do wartości rzeczywistych zależy od wylosowanych wag, liczby epoch oraz współczynnika uczenia.    

