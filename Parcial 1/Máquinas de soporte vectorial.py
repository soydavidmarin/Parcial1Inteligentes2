import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn import metrics
import seaborn as sb
from sklearn.metrics import precision_recall_fscore_support
from sklearn.svm import SVC
from sklearn.feature_selection import SequentialFeatureSelector

import time

sfs = None
selectedFeatures = []
modelo = SVC()


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


def partition(test_size):
  global XTrain, XTest, yTrain, yTest
  XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=test_size)


inicio = time.time()

#Subida de datasets
dataframe = pd.read_csv('bank-full.csv', delimiter=";")
dataframe.head(7)

#Conversión de valores categoricos a numericos
columnas = dataframe.columns.values
print('numero de columnas ', len(columnas))
for columnaActual in columnas:
  if dataframe[columnaActual].dtypes == object or dataframe[columnaActual].dtypes == bool:
    dataframe[columnaActual] = LabelEncoder(
    ).fit_transform(dataframe[columnaActual])

print(dataframe.head(7))

#Particion del dataset
XFull = dataframe.drop(["y"], axis=1)
X = dataframe.drop(["y"], axis=1)
y = dataframe["y"]
XTrain, XTest, yTrain, yTest = None, None, None, None
partition(0.1)
sequentialFeatureSelector(5, "accuracy", 5)
dropFeaturesNoSelected()

modelo.fit(XTrain, yTrain)
yPredictSVC = modelo.predict(XTest)
print("Accruracy =", metrics.accuracy_score(yTest, yPredictSVC))

matrizSVC = confusion_matrix(yTest, yPredictSVC)
print(matrizSVC)
sb.heatmap(matrizSVC, annot=True, cmap="Blues")

precision_recall_fscore_support(yTest, yPredictSVC, average=None)
etiquetas = ["no", "si"]
print(classification_report(yTest, yPredictSVC, target_names=etiquetas))
print("Accuracy=", modelo.score(XTest, yTest))
