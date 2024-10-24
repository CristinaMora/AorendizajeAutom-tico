import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def cleanData(data):
    # TO-DO clean the dataframe. Return the dataframe cleaned.

    data["score"] = pd.to_numeric(data["score"], errors='coerce')
    data["user score"] = pd.to_numeric(data["user score"], errors='coerce')
    data["score"] = data["score"].astype(np.float64)
    data["user score"] = data["user score"].astype(np.float64)
    data["score"] = data["score"] / 10
    data = data.dropna(subset=["score", "user score"])
    return data


def load_data_csv(path, x_colum, y_colum):
    data = pd.read_csv(path)
    data = cleanData(data)
    X = data[x_colum].to_numpy()
    Y = data[y_colum].to_numpy()

    print('X: ', X)
    print('Y: ', Y)
    # Crear gráfico de línea
    plt.plot(X, Y, "bo", markersize=1)
    # Añadir etiquetas y título
    plt.xlabel('Eje X')
    plt.ylabel('Eje Y')
    plt.title('Gráfico de Línea')

    # Mostrar gráfico
    plt.show()
    return X, Y

