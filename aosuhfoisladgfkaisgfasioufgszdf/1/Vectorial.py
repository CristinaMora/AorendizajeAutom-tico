import numpy as np
import copy
import math


class LinearReg2:
    """
    Computes the cost function for linear regression.

    Args:
        x (ndarray): Shape (m,) Input to the model
        y (ndarray): Shape (m,) the real values of the prediction
        w, b (scalar): Parameters of the model
    """

    def __init__(self, x, y, w, b):
        # (scalar): Parameters of the model
        self.x = x
        self.y = y
        self.w = w
        self.b = b

    """
    Computes the linear regression function.

    Args:
        x (ndarray): Shape (m,) Input to the model

    Returns:
        the linear regression value
    """

    def f_w_b(self, x):
        return np.dot(self.x,self.w) + self.b

    """
    Computes the cost function for linear regression.

    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points
    """

    def compute_cost(self):
        sum = 0
        sum =np.sum((self.y - self.f_w_b(self.x)) ** 2)
        total_cost = sum / (2 * len(self.y))
        return total_cost

    """
    Computes the gradient for linear regression 
    Args:

    Returns
      dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
      dj_db (scalar): The gradient of the cost w.r.t. the parameter b     
     """

    def compute_gradient(self):
        dj_dw = 0
        dj_db = 0
        sum = 0
        sum = np.sum((self.f_w_b(self.x) - self.y) * self.x)
        
        dj_dw = sum / (len(self.y))
        
        sum = 0

        sum = np.sum(self.f_w_b(self.x) - self.y)
        dj_db = sum / (len(self.y))

        return dj_dw, dj_db

    """
    Performs batch gradient descent to learn theta. Updates theta by taking 
    num_iters gradient steps with learning rate alpha

    Args:
      alpha : (float) Learning rate
      num_iters : (int) number of iterations to run gradient descent
    Returns
      w : (ndarray): Shape (1,) Updated values of parameters of the model after
          running gradient descent
      b : (scalar) Updated value of parameter of the model after
          running gradient descent
      J_history : (ndarray): Shape (num_iters,) J at each iteration,
          primarily for graphing later
      w_initial : (ndarray): Shape (1,) initial w value before running gradient descent
      b_initial : (scalar) initial b value before running gradient descent
    """

    def gradient_descent(self, alpha, num_iters):
        # An array to store cost J and w's at each iteration — primarily for graphing later
        J_history = []
        w_history = []
        for i in range(num_iters):
            j = self.compute_cost()
            J_history.append(j)
            self.w = np.dot(-alpha * self.compute_gradient().dj_dw)
            self.b = np.dot(- alpha * self.compute_gradient().dj_db)

        w_initial = copy.deepcopy(self.w)  # avoid modifying global w within function
        b_initial = copy.deepcopy(self.b)  # avoid modifying global b within function

        return self.w, self.b, J_history, w_initial, b_initial


def cost_test_obj2(x, y, w_init, b_init):
    lr = LinearReg2(x, y, w_init, b_init)
    cost = lr.compute_cost()
    return cost


def compute_gradient_obj2(x, y, w_init, b_init):
    lr = LinearReg2(x, y, w_init, b_init)
    dw, db = lr.compute_gradient()
    return dw, db
