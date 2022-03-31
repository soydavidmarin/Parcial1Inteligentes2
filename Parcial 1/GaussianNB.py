import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.feature_selection import SequentialFeatureSelector

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
for columnaActual in columnas:  
  if dataframe[columnaActual].dtypes == object or dataframe[columnaActual].dtypes == bool:    
    dataframe[columnaActual]=LabelEncoder().fit_transform(dataframe[columnaActual])        

print(dataframe.head(7))

#print(dataframe['balance'].max())
#print(dataframe['balance'].min())

'''
count = 0
for i in dataframe['balance']:
    if i < 0:
        print(i)
        count+=1
print("Total: ", count)
'''

#Particion del dataset
X=dataframe.drop(["y"],axis=1)
y=dataframe["y"]
XTrain,XTest,yTrain,yTest = None,None,None,None


def partition(test_size):    
    global XTrain,XTest,yTrain,yTest
    XTrain,XTest,yTrain,yTest=train_test_split(X,y,test_size=test_size)

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

Gaussian Naive Bayes
"""
yPredictGNB = None
modeloGNB=    ()
def trainGNB():    
    global yPredictGNB
    modeloGNB.fit(XTrain,yTrain)
    yPredictGNB=modeloGNB.predict(XTest)   
    
sfs = None    
selectedFeatures = []
def sequentialFeatureSelector(features, modelo):
    global sfs
    sfs = SequentialFeatureSelector(modelo, n_features_to_select=features)
    sfs.fit(X, y)                
    print("Caracteristicas seleccionadas:")
    features = sfs.get_support();
    for i in range(0, len(features)):
        if features[i]:
            selectedFeatures.append(columnas[i])
    print(selectedFeatures)
    
    
partition(0.3)
sequentialFeatureSelector(5,modeloGNB)
#trainGNB() 
'''

trainGNB()
matrizGNB=confusion_matrix(yTest,yPredictGNB)
print('--Matriz de confusión--')
print(matrizGNB)
etiquetas=["no","si"]
print(classification_report(yTest,yPredictGNB,target_names=etiquetas))
print("Accuracy=",modeloGNB.score(XTest,yTest))
'''