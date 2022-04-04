# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 14:07:50 2022

@author: Richa
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.neighbors import KNeighborsClassifier
import time
from sklearn import metrics
inicio = time.time()

# Subida de datasets
dataframe = pd.read_csv('bank-full.csv', delimiter=";")
dataframe.head(7)


# Estadisticas
totalData = len(dataframe.index)
print(dataframe.groupby("y").size())
print("%No: ", (dataframe.groupby("y").size()["no"]/totalData)*100, "%")
print("%Si: ", (dataframe.groupby("y").size()["yes"]/totalData)*100, "%")
print("Total: ", totalData)


# Conversión de valores categoricos a numericos
columnas = dataframe.columns.values
print('numero de columnas ', len(columnas))
for columnaActual in columnas:
    if dataframe[columnaActual].dtypes == object or dataframe[columnaActual].dtypes == bool:
        dataframe[columnaActual] = LabelEncoder(
        ).fit_transform(dataframe[columnaActual])

print(dataframe.head(7))


# Particion del dataset
XFull = dataframe.drop(["y"], axis=1)
X = dataframe.drop(["y"], axis=1)
y = dataframe["y"]
XTrain, XTest, yTrain, yTest = None, None, None, None

#
yPredict = None
modelo = KNeighborsClassifier(n_neighbors=77)


def partition(test_size, random_state):
    global XTrain, XTest, yTrain, yTest
    XTrain, XTest, yTrain, yTest = train_test_split(
        X, y, test_size=test_size, random_state=random_state)


def normalitationRobust():
    global XTrain, XTest
    escalar = RobustScaler()
    XTrain = escalar.fit_transform(XTrain)
    XTest = escalar.transform(XTest)


def normalitationMinMax():
    global XTrain, XTest
    escalar = MinMaxScaler()
    XTrain = escalar.fit_transform(XTrain)
    XTest = escalar.transform(XTest)


"""Entrenamiento de modelos

KNN
"""


def train():
    global yPredict
    modelo.fit(XTrain, yTrain)
    yPredict = modelo.predict(XTest)


sfs = None
selectedFeatures = []


def sequentialFeatureSelector(featToSel, scoring, cv):
    global sfs, selectedFeatures, X, modelo
    X = XFull
    selectedFeatures.clear()
    sfs = SequentialFeatureSelector(
        modelo, n_features_to_select=featToSel, scoring=scoring, cv=cv)
    sfs.fit(X, y)
    features = sfs.get_support()
    for i in range(0, len(features)):
        if features[i]:
            selectedFeatures.append(columnas[i])
    # print("Caracteristicas seleccionadas:")
    # print(selectedFeatures)


def dropFeaturesNoSelected():
    global X
    X = XFull
    colTemp = []
    for i in columnas:
        if i not in selectedFeatures:
            colTemp.append(i)
    X = dataframe.drop(colTemp, axis=1)


def metricas():
    matriz = confusion_matrix(yTest, yPredict)
    print('--Matriz de confusión--')
    print(matriz)
    etiquetas = ["no", "si"]
    print(classification_report(yTest, yPredict, target_names=etiquetas))
    print("Accuracy=", modelo.score(XTest, yTest))



partition(0.1, 5)
normalitationMinMax()
sequentialFeatureSelector(5, "accuracy", 5)
dropFeaturesNoSelected()
train()
acc = modelo.score(XTest, yTest)

print("Accruracy =",metrics.accuracy_score(yTest,yPredict))
matrizSVC=confusion_matrix(yTest,yPredict)
metricas()
fin = time.time()
print("time: ", (fin-inicio)/60, 'min')
