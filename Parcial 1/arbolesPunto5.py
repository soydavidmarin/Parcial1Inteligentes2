# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 15:11:27 2022

@author: Richa
"""

import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.feature_selection import SequentialFeatureSelector

import time
inicio = time.time()

# Subida de datasets
dataframe = pd.read_csv('bank-full.csv', delimiter=";")
dataframe.head(7)

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

#Variables
modelo = None
max = 0
imax = 0
partitions = [0.1]
sfs = None
selectedFeatures = []
featuresToSelect = [3, 5, len(columnas)-2]
score = ["accuracy", "recall", "f1"]
cv = [5, 7, 9]

def partition(test_size):
    global XTrain, XTest, yTrain, yTestF, yTest
    XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=test_size)

def dropFeaturesNoSelected():
    global X
    X = XFull
    colTemp = []
    for i in columnas:
        if i not in selectedFeatures:
            colTemp.append(i)
    X=dataframe.drop(colTemp,axis=1)
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
"""Entrenamiento de modelos

Arboles de desición
"""
bestConf={
    "features": 0,
    "score": 0,
    "cv": 0, 
    "deep": 0
    }
print("iMax:", imax, " Max:", max)
print(bestConf)
partition(0.1)
modelo = DecisionTreeClassifier(max_depth=7)
# sequentialFeatureSelector(bestConf["features"], bestConf["score"], bestConf["cv"])
sequentialFeatureSelector(3, "accuracy", 9)
dropFeaturesNoSelected()
modelo.fit(XTrain, yTrain)
yPredictDTC = modelo.predict(XTest)


print("Accuracy ", metrics.accuracy_score(yTest, yPredictDTC))

matrizDTC = confusion_matrix(yTest, yPredictDTC)
print(matrizDTC)

  

fin = time.time()
print("time: ", (fin-inicio)/60, 'min')
