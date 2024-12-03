import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import load_data, load_weights,one_hot_encoding, accuracy,compute_metrics
from MLP import MLP
from public_test import compute_cost_test, predict_test

x,y = load_data('data/ex3data1.mat')
theta1, theta2 = load_weights('data/ex3weights.mat')

_mlp = MLP(theta1,theta2)
a1,a2,a3,z2,z3 = _mlp.feedforward(x)

prediction = _mlp.predict(a3)

predict_test(prediction,y,accuracy)

compute_cost_test(_mlp,a3, one_hot_encoding(y))


metrics = compute_metrics(y, prediction, positive_class=0)

print("Matriz de Confusión:")
print(f"Precision: {metrics['precision']:.2f}")
print(f"Recall: {metrics['recall']:.2f}")
print(f"F1-Score: {metrics['f1_score']:.2f}")

print(f"TP: {metrics['TP']}, FP: {metrics['FP']}, FN: {metrics['FN']}, TN: {metrics['TN']}")

