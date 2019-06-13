import csv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn import preprocessing

import math
import matplotlib.pyplot as plt
import numpy as np

import operator

def load_data(filename):
    d = csv.reader(open(filename))

    x = []
    y = []

    for line in d:
        x.append([float(line[0]), float(line[1])])
        y.append(int(line[2]))

    return x, y

def run():

    # x recebe um array de 2 posicoes: [0] -> altura do(a) peao/peoa, [1] -> peso do(a) sujeito/sujeita
    # y recebe 0 ou 1: 0 -> omi e 1 -> muie
    x, y = load_data("alturapeso.csv")

    #labelEncoder
    """ le = preprocessing.LabelEncoder()
    le.fit(x[1])

    x[1] = le.transform(x[1])

    #OneHotEncoder
    lx = preprocessing.OneHotEncoder()
    lx.fit(x[1])

    x[1] = lx.transform(x[1]) """

    # normalização dos dados 
    """ value = preprocessing.StandardScaler()
    
    x = value.fit_transform(x) """

    # we know about that and it's just work!! 
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)

    # parametro de entrada do KNN: neste caso os 7 vizinhos mais proximos
    obj = KNeighborsClassifier(n_neighbors = 9)

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

    # meninas vestem rosa, meninos vestem azul - brincadeira a parte
    plt.plot(x_real_muie, y_real_muie , marker='+', linestyle='none', markersize=5.5, color='#F073EC')
    plt.plot(x_real_omi, y_real_omi , marker='.', linestyle='none', markersize=3.5, color='#50B5EF')
        
    plt.ylabel('peso')
    plt.xlabel('altura')

    # avalia este novo conjunto de valore baseado no treino feito anteriormente para classificar
    res = obj.predict(x_test)

    new = [1.74, 70]

    #new = value.transform([new])

    #print(new[0][1])

    # avalia apenas um novo individuo se e omi ou muie
    res2 = obj.predict([new])
    #res2 = obj.predict(new)

    #print('\nxtest: ', x_test)

    print('-----------------------------------------------------')
    #print(res)
    print(res2[0]) 

    #plt.plot(new[0][0], new[0][1] , marker='s', linestyle='none', markersize=4.5, color='green')
    plt.plot(new[0], new[1] , marker='s', linestyle='none', markersize=4.5, color='green')
    #plt.annotate('new individual', xy=(new[0][0], new[0][1]), xytext=(0.80, 0.75), arrowprops=dict(facecolor='#777777', shrink=0.25, width=1.3, headwidth=8, headlength=5))
    plt.annotate('new individual', xy=(new[0], new[1]), xytext=(1.80, 75), arrowprops=dict(facecolor='#777777', shrink=0.25, width=1.3, headwidth=8, headlength=5))

    if res2[0] == 1:
        print('The new individial is a woman')
    else:
        print('The new individual is a man')

    accuracy = accuracy_score(y_test, res)
    print('acuracia: ', accuracy)
    plt.show()
    
    return accuracy


if __name__ == "__main__":
    
    acc = [] 
    repeat = []

    for i in range(1):
        acc.append(run())
        repeat.append(i)
    
    """ fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_ylim([0, 1])
    plt.plot(repeat, acc, color='#6073EC')
    
    plt.savefig('normalize.png')
    plt.show() """
    

    #obs:
    # Normalização melhorou a acurácia da classificação dos dados 
    # Onehotencoder foi quem pior fez efeito
    # LabelEncoder não mudou o efeito
    # a saida do sujeito proposto teve mudanças com cada modo 
    #   - com a normalização o fator 'man' tornou-se quase constante
    #   - quanto que normal era representando em suma por 'woman'