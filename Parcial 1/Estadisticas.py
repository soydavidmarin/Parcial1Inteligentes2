
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sb
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn import metrics


from scipy.stats import shapiro
from scipy.stats import normaltest
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import graphviz

from sklearn.svm import SVC

from google.colab import files
import io


#Subida de datasets
archivo=files.upload()
archivo2=files.upload()
dataframe=pd.read_csv(io.BytesIO(archivo['bank-full.csv']),delimiter=";")
dataframeLite=pd.read_csv(io.BytesIO(archivo2['bank.csv']),delimiter=";")
dataframe.head(7)

#Estadisticas 
totalData = len(dataframe.index)
print(dataframe.groupby("y").size())
print("%No: ",(dataframe.groupby("y").size()["no"]/totalData)*100,"%")
print("%Si: ",(dataframe.groupby("y").size()["yes"]/totalData)*100,"%")
print("Total: ", totalData)

#Eliminación de columnas posiblemente inutiles (Elegidas por los autores)
dataframeLite=dataframeLite.drop(["contact", "pdays"],axis=1)
dataframe=dataframe.drop(["contact", "pdays"],axis=1)
dataframe.head(7)

#Columnas con distribución normal
columnas=dataframe.columns.values
normal=[]
noNormal=[]
for columnaActual in columnas:
    #Conversión de valores categoricos a numericos
  if dataframe[columnaActual].dtypes == object or dataframe[columnaActual].dtypes == bool:    
    dataframe[columnaActual]=LabelEncoder().fit_transform(dataframe[columnaActual])    
    dataframeLite[columnaActual]=LabelEncoder().fit_transform(dataframeLite[columnaActual])
 
  datosColumna=dataframeLite[columnaActual]
  #print(datosColumna.shape)
  stat,p=shapiro(datosColumna)
  #print("stat=",stat," p=",p)
  if p>0.05:
    normal.append(columnaActual)
  else:
    noNormal.append(columnaActual)
  #print("--------------")
print("Con distribucion normal: ",normal)
print("Sin distribucion normal: ",noNormal)
dataframe.head(7)

#Matriz de correlación
colormap = plt.cm.coolwarm
plt.figure(figsize=(12,12))
plt.title('Matriz de correlación', y=1.05, size=15)
sb.heatmap(dataframe.drop(['y'], axis=1).astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)

#Graficas de las distribuciones
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

dataframe.drop(["y"],axis=1).hist()
plt.show()


#Histogramas
columnas=list(dataframe.drop(["y"],axis=1).columns)
for columnaActual in columnas:
  no = dataframe[dataframe["y"] == 0]
  si = dataframe[dataframe["y"] == 1]

  columnaAnalizadaVender=no[columnaActual].to_numpy()
  columnaAnalizadaComprar=si[columnaActual].to_numpy()
  
  clases = ['no', 'si']

  sb.distplot(columnaAnalizadaVender, hist = True, kde = True, kde_kws = {'linewidth': 3}, label = clases[0])
  sb.distplot(columnaAnalizadaComprar, hist = True, kde = True, kde_kws = {'linewidth': 3}, label = clases[1])
      
  # Plot formatting
  plt.legend(prop={'size': 16}, title = 'Categorias')
  plt.title('Distribution Plot '+columnaActual)
  plt.xlabel('Data')
  plt.ylabel('Distribution')
  plt.show()

