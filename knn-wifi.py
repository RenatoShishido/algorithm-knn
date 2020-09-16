import subprocess
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from threading import Timer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def signal_wifi():
    redes = ["*BOLANDO", "Marmitteiros2", "Marmitteiros2", "MARMITTEIROS", "Lemos", "CENTRO ELETRONICO", "Top Espetos Clientes",
             "CONVENIENCIA-EMPORIO", "SKY_B8A275", "Arya", "CONVENIENCIA-EMPORIO-5G", "TOPESPETOS", "VIVO-6C58"]
    essid = []
    essid_final = []
    test = []
    sinal_s = []
    sinal = []
    sinal_final = []
    aux = []

    # Pegando os valores do wifi pelo terminal
    var = subprocess.getoutput("nmcli dev wifi")
    for line in var.splitlines():
        test.append("".join(line.split()))

    # Tirando as barrinha de sinal e splitando por Infra para pegar o ESSID
    for string in test:
        barra = "/".join(string.split('▂'))
        sinal_s.append(barra.split('/'))
        essid.append(barra.split('Infra'))

    # Deletando os primeiros elementos pelo fato que sempre sera o nome
    # EX: [Signal, freq, etc..]
    del sinal_s[0]
    del essid[0]

    # Inserindo os ESSID no array final
    for i in essid:
        essid_final.append(i[0])

    # Inserir os sinais de cada ESSID
    for i in sinal_s:
        sinal.append(i[1])

    # Tirando o 's' de cada sinal
    for string in sinal:
        sinal_final.append("".join(string.split('s')))

    # Inserindo os dados de sinal dentro do array final
    count = 0
    count2 = 0
    for x in redes:
        if x in essid_final:
            aux.append(sinal_final[count])
            count = count + 1
        else:
            aux.append(0)
            count2 = count + 1

    return aux


def knn_wifi(dados):
    # Treino
    train_data = pd.read_csv("train.csv")
    test_data = pd.read_csv("test.csv")

    y = train_data["Class"]
    train_data.drop(["Class"], axis=1, inplace=True)
    X = train_data.values

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42)

    model = KNeighborsClassifier(n_neighbors=1, weights="distance")
    model.fit(X_train, y_train)
    pred = model.predict(X_val)

    # Teste
    X_test = test_data.values

    test_pred = model.predict(X_test)
    output = pd.DataFrame({'Saida': test_pred})

    # Predicao

    prediga = model.predict([dados])
    saida = pd.DataFrame({'predict': prediga})
    print(saida)


def disparador():
    knn_wifi(signal_wifi())
    Timer(5.0, disparador).start()


disparador()
