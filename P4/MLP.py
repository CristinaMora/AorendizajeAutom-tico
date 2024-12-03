import numpy as np

class MLP:

    """
    Constructor: Computes MLP.

    Args:
        theta1 (array_like): Weights for the first layer in the neural network.
        theta2 (array_like): Weights for the second layer in the neural network.
    """
    def __init__(self,theta1,theta2):
        self.theta1 = theta1
        self.theta2 = theta2
        
    """
    Num elements in the training data. (private)

    Args:
        x (array_like): input data. 
    """
    def _size(self,x):
        return x.shape[0]
    
    """
    Computes de sigmoid function of z (private)

    Args:
        z (array_like): activation signal received by the layer.
    """
    def _sigmoid(self,z):
      return 1 / (1 + np.power(np.e, -z))

    """
    Run the feedwordwar neural network step

    Args:
        z (array_like): activation signal received by the layer.

	Return 
	------
	a1,a2,a3 (array_like): activation functions of each layers
    z2,z3 (array_like): signal fuction of two last layers
    """
    def feedforward(self,x):
        m =self._size(x)
        
        a1 = np.hstack([np.ones((m, 1)), x])

     
        z2 = a1 @ self.theta1.T
        a2 = np.hstack([np.ones((m, 1)), self._sigmoid(z2)])

        z3 = a2 @ self.theta2.T
        a3 = self._sigmoid(z3)

        return a1, a2, a3, z2, z3
    """
    Computes only the cost of a previously generated output (private)

    Args:
        yPrime (array_like): output generated by neural network.
        y (array_like): output from the dataset

	Return 
	------
	J (scalar): the cost.
    """
    def compute_cost(self, yPrime,y): # calcula solo el coste, para no ejecutar nuevamente el feedforward.
        m = self._size(yPrime)
        J = -np.sum(np.sum(y * np.log(yPrime) + (1 - y) * np.log(1 - yPrime)))/m
        return J
    
    """
    Get the class with highest activation value

    Args:
        a3 (array_like): output generated by neural network.

	Return 
	------
	p (scalar): the class index with the highest activation value.
    """
    def predict(self,a3):
        
        return np.argmax(a3 ,axis = 1) 

    