import numpy as np
import pandas as pd


def cleanData(data):
   #TO-DO clean the dataframe. Return the dataframe cleaned.
    data = data.dropna(subset=["score","user score"])
    data ["score","user score"] = data ["score","user score"].astype(np.float32)
    return  data.dropna(subset=["score","user score"])

def load_data_csv(path,x_colum,y_colum):
    data = pd.read_csv(path)
    data = cleanData(data)
    X = data[x_colum].to_numpy()
    y = data[y_colum].to_numpy()
    return X, y