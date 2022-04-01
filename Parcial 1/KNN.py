import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.neighbors import KNeighborsClassifier
import time
inicio = time.time()

#Subida de datasets
dataframe=pd.read_csv('bank-full.csv',delimiter=";")
dataframe.head(7)


#Estadisticas 
totalData = len(dataframe.index)
print(dataframe.groupby("y").size())
print("%No: ",(dataframe.groupby("y").size()["no"]/totalData)*100,"%")
print("%Si: ",(dataframe.groupby("y").size()["yes"]/totalData)*100,"%")
print("Total: ", totalData)


#Conversión de valores categoricos a numericos
columnas=dataframe.columns.values
print('numero de columnas ',len(columnas))
for columnaActual in columnas:  
  if dataframe[columnaActual].dtypes == object or dataframe[columnaActual].dtypes == bool:    
    dataframe[columnaActual]=LabelEncoder().fit_transform(dataframe[columnaActual])        

print(dataframe.head(7))


'''
count = 0
for i in dataframe['balance']:
    if i < 0:
        print(i)
        count+=1
print("Total: ", count)
'''

#Particion del dataset
XFull=dataframe.drop(["y"],axis=1)
X=dataframe.drop(["y"],axis=1)
y=dataframe["y"]
XTrain,XTest,yTrain,yTest = None,None,None,None

#
yPredict = None
modelo = KNeighborsClassifier(n_neighbors=77)

def partition(test_size,random_state):    
    global XTrain,XTest,yTrain,yTest
    XTrain,XTest,yTrain,yTest=train_test_split(X,y,test_size=test_size,random_state=random_state)

def normalitationRobust():    
    global XTrain,XTest
    escalar=RobustScaler()
    XTrain=escalar.fit_transform(XTrain)
    XTest=escalar.transform(XTest)
    
def normalitationMinMax():    
    global XTrain,XTest
    escalar=MinMaxScaler()
    XTrain=escalar.fit_transform(XTrain)
    XTest=escalar.transform(XTest)

"""Entrenamiento de modelos

KNN
"""

def train():    
    global yPredict
    modelo.fit(XTrain,yTrain)
    yPredict=modelo.predict(XTest)   
    
sfs = None    
selectedFeatures = []
def sequentialFeatureSelector(featToSel,scoring,cv):
    global sfs, selectedFeatures,X,modelo
    X = XFull
    selectedFeatures.clear()
    sfs = SequentialFeatureSelector(modelo, n_features_to_select=featToSel, scoring=scoring, cv = cv)
    sfs.fit(X, y)                    
    features = sfs.get_support();
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
    X=dataframe.drop(colTemp,axis=1)    
    
def metricas():
    matriz=confusion_matrix(yTest,yPredict)
    print('--Matriz de confusión--')
    print(matriz)
    etiquetas=["no","si"]
    print(classification_report(yTest,yPredict,target_names=etiquetas))
    print("Accuracy=",modelo.score(XTest,yTest))    
    

test = [0.1,0.2,0.3]
featuresToSelect = [3,5,len(columnas)-2]                
randomState = [5,7,9]
score=["accuracy","recall","f1"]
cv=[5,7,9]
maxAcc = 0
bestConf={
    "Test": 0,
    "Features": 0,
    "RandomState": 0,
    "score": 0,
    "cv": 0
    }
for t in test:
    for f in featuresToSelect:     
        for r in randomState:
            for s in score:
                for c in cv:                                     
                    partition(t, r)
                    # normalitationMinMax()
                    sequentialFeatureSelector(f,s,c)
                    dropFeaturesNoSelected()
                    train() 
                    acc = modelo.score(XTest,yTest)                    
                    print(t,f,r,s,c,acc)
                    if acc > maxAcc:
                        maxAcc = acc
                        bestConf["Test"]=t
                        bestConf["Features"]=f
                        bestConf["RandomState"]=r
                        bestConf["score"]=s
                        bestConf["cv"]=c
print("Best:")
print(bestConf)
print(maxAcc*100,"%")        


fin = time.time()
print("time: ",(fin-inicio)/60,'min')