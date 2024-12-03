import numpy as np
from matplotlib import pyplot
from scipy.io import loadmat
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
"""
Load data from the dataset.
"""
def load_data(file):
    data = loadmat(file, squeeze_me=True)
    x = data['X']
    y = data['y']
    return x,y

"""
Load weights from the weights file 
"""
def load_weights(file):
    weights = loadmat(file)
    theta1, theta2 = weights['Theta1'], weights['Theta2']
    return theta1, theta2


"""
Implementation of the one hot encoding... You must use OneHotEncoder function of the sklern library. 
Probably need to use reshape(-1, 1) to change size of the data
"""
def one_hot_encoding(Y):
    oneHotEncoder = OneHotEncoder()
    YEnc = oneHotEncoder.fit(Y.reshape(-1,1))
    return YEnc.transform(Y.reshape(-1,1)).toarray()

"""
Implementation of the accuracy metrics function
"""
def accuracy(P,Y):
	return np.mean(P == Y)

def compute_metrics(y_true, y_pred, positive_class=0):
    """
    Calcula la matriz de confusión, precisión, recall y F1-score para la clase positiva.

    Parámetros:
    - y_true: np.ndarray, etiquetas reales (1D).
    - y_pred: np.ndarray, etiquetas predichas (1D).
    - positive_class: int, la clase que consideramos como positiva (por defecto 0).

    Retorna:
    - metrics: dict, contiene la matriz de confusión, precisión, recall y F1-score.
    """
    # Binarizar las etiquetas para la clase positiva
    

    TP = np.sum((y_pred == y_true) & (y_true == 0))
    FP = np.sum((y_pred == 0) & (y_true != 0))
    FN = np.sum((y_pred != 0) & (y_true == 0))
    TN = np.sum((y_pred != 0) & (y_true != 0))



    # Calcular métricas
    precision = TP/(TP+FP)
    recall = TP/(TP/FN)
    f1 =2*(precision * recall / (precision + recall))

    # Retornar resultados
    metrics = {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "TP": TP,
    }
    return metrics