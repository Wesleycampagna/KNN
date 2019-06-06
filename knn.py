import csv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV

import matplotlib.pyplot as plt

import operator

def load_data(filename):
    d = csv.reader(open(filename))

    x = []
    y = []

    for line in d:
        x.append([float(line[0]), float(line[1])])
        y.append(int(line[2]))

    return x, y

if __name__ == "__main__":

    # x recebe um array de 2 posicoes: [0] -> altura do(a) peao/peoa, [1] -> peso do(a) sujeito/sujeita
    # y recebe 0 ou 1: 0 -> omi e 1 -> muie
    x, y = load_data("alturapeso.csv")

    # we don't know about that, just work!! 
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)

    # parametro de entrada do KNN: neste caso os 7 vizinhos mais proximos
    obj = KNeighborsClassifier(n_neighbors = 7)

    #treinamento do cojunto de teste 
    obj.fit(x_train, y_train)

    x_real_muie = []
    x_real_omi = []
    y_real_muie = []
    y_real_omi = []

    for i in range(len(x)):
        if y[i] == 1:
            x_real_muie.append(x[i][0])
            y_real_muie.append(x[i][1])
        else:
            x_real_omi.append(x[i][0])
            y_real_omi.append(x[i][1])

    # meninas vestem rosa, meninos vestem azul
    plt.plot(x_real_muie, y_real_muie , marker='+', linestyle='none', markersize=5.5, color='#F073EC')
    plt.plot(x_real_omi, y_real_omi , marker='.', linestyle='none', markersize=3.5, color='#50B5EF')
        
    plt.ylabel('peso')
    plt.xlabel('altura')

    # avalia este novo conjunto de valore baseado no treino feito anteriormente para classificar
    res = obj.predict(x_test)

    new = [1.74, 70]
    # avalia apenas um novo individuo se e omi ou muie
    res2 = obj.predict([new])

    #print('\nxtest: ', x_test)

    print('-----------------------------------------------------')
    #print(res)
    print(res2[0])

    plt.plot(new[0], new[1] , marker='s', linestyle='none', markersize=4.5, color='green')
    plt.annotate('new individual', xy=(new[0], new[1]), xytext=(1.80, 75), arrowprops=dict(facecolor='#777777', shrink=0.25, width=1.3, headwidth=8, headlength=5))

    if res2[0] == 1:
        print('The new individial is a woman')
    else:
        print('The new individual is a man')

    print('acuracia: ', accuracy_score(y_test, res))

    plt.show()