import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def cleanData(data):
   #TO-DO clean the dataframe. Return the dataframe cleaned.
    data = data.dropna(subset=["score","user score"])
    data["score"] = pd.to_numeric(data["score"], errors='coerce')
    data["user score"] = pd.to_numeric(data["user score"], errors='coerce')
    data ["score"] = data ["score"].astype(np.float32)
    data ["user score"] = data ["user score"].astype(np.float32)
    data["score"] = data["score"] / 10
    return  data

def load_data_csv(path,x_colum,y_colum):
    data = pd.read_csv(path)
    #Limpiamos la tabla de elementos que no sean numeros y ponemos todos los datos a la misma escala
    data = cleanData(data)
    #Creamos las columnas segun los datos pedidos
    X = data[x_colum].to_numpy()
    Y = data[y_colum].to_numpy()
    # Crear gráfico de puntos, x ,y, tipo puntos, tamaño de los puntos
    plt.plot(X, Y, "bo", markersize = 1)
    # Añadir etiquetas y título
    plt.xlabel('Eje X')
    plt.ylabel('Eje Y')
    plt.title('Gráfico')
    plt.show()
    return X, y
