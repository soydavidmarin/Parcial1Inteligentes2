# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 14:23:34 2022

@author: Richa
"""

import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


dataframe = pd.read_csv('bank-full.csv', delimiter=";")
# Conversión de valores categoricos a numericos
columnas = dataframe.columns.values
print('numero de columnas ', len(columnas))
for columnaActual in columnas:
    if dataframe[columnaActual].dtypes == object or dataframe[columnaActual].dtypes == bool:
        dataframe[columnaActual] = LabelEncoder(
        ).fit_transform(dataframe[columnaActual])
X = dataframe.drop(["y"], axis=1)
y = dataframe["y"]
XTrain, XTest, yTrain, yTest = None, None, None, None

#
yPredict = None

def metricas():
    matriz = confusion_matrix(yTest, yPredict)
    print('--Matriz de confusión--')
    print(matriz)
    etiquetas = ["no", "si"]
    print(classification_report(yTest, yPredict, target_names=etiquetas))
    

def partition(test_size, random_state):
    global XTrain, XTest, yTrain, yTest, X, y
    XTrain, XTest, yTrain, yTest = train_test_split(
        X, y, test_size=test_size, random_state=random_state)



partition(0.1, 5)

modelo = pickle.load(open("knn-model.sav", 'rb'))
yPredict = modelo.predict(XTest)
metricas()