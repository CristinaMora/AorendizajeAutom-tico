import numpy as np
import copy
import math

from LinearRegressionMulti import LinearRegMulti

class LogisticRegMulti(LinearRegMulti):

    """
    Computes the cost function for linear regression.

    Args:
        x (ndarray): Shape (m,) Input to the model
        y (ndarray): Shape (m,) the real values of the prediction
        w, b (scalar): Parameters of the model
        lambda: Regularization parameter. Most be between 0..1. 
        Determinate the weight of the regularization.
    """
    def __init__(self, x, y,w,b, lambda_):
        super().__init__(x, y,w,b,lambda_)


    """
    Computes the linear regression function.

    Args:
        x (ndarray): Shape (m,) Input to the model
    
    Returns:
        the linear regression value
    """
    def f_w_b(self, x):
        return 1 / (1 + np.exp(-(x @ self.w + self.b)))
    
    
    
    """
    Computes the cost function for linear regression.

    Returns
        total_cost (float): The cost of using w, b as the parameters for linear regression
               to fit the data points in x and y
    """
    def compute_cost(self):
        
        #nEjmeplos
        m = len(self.x)
        
        #Calcula las predicciones del modelo
        y = self.f_w_b(self.x)

        #Nos definimos un valor muy pequeño para evitar problemas de calculo cuando y sea 0 o 1, ya que el algoritmo en 0 es indefinido
        epsilon = 1e-10

        # Ajustamos el rango de y para que todos sean valores validos para el algoritmo
        y = np.clip(y, epsilon, 1 - epsilon)

        #Calculo del Loss
        total_cost = (-1 / m) * np.sum(self.y * np.log(y) + (1 - self.y) * np.log(1 - y))
        reg_cost = self._regularizationL2Cost()

        #Suma de costes
        return total_cost + reg_cost
    
    """
    Computes the gradient for linear regression 
    Args:

    Returns
      dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
      dj_db (scalar): The gradient of the cost w.r.t. the parameter b     
    """
    def compute_gradient(self):
        """
        Calcula los gradientes de w y b para regresión multivariable.
        """
        

        return super().compute_gradient()
  
    
def cost_test_multi_obj(x,y,w_init,b_init):
    lr = LogisticRegMulti(x,y,w_init,b_init,0)
    cost = lr.compute_cost()
    return cost

def compute_gradient_multi_obj(x,y,w_init,b_init):
    lr = LogisticRegMulti(x,y,w_init,b_init,0)
    dw,db = lr.compute_gradient()
    return dw,db
