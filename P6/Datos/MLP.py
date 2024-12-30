import numpy as np
import math

class MLP:

    """
    Constructor: Computes MLP.

    Args:
        inputLayer (int): size of input
        hiddenLayers (array like): hidden layers of the model: len (hiddenLayers) -> number of hidden layers, hiddenLayers[i] -> size of the layer
        outputLayer (int): size of output layer
        layers (int): number of hidden layers
        seed (scalar): seed of the random numeric.
        epsilom (scalar) : random initialization range. e.j: 1 = [-1..1], 2 = [-2,2]...
    """

    def __init__(self, inputLayer, hiddenLayers, outputLayer, seed=0, epsilon = 0.12):
        np.random.seed(seed)
        
        #Necesitamos una capa de input, n layers ocultas y una capa de output
        self.input = inputLayer
        self.output = outputLayer
        self.hiddenLayers = hiddenLayers        # Recibimos las capas ocultas
        self.nlayers = len(hiddenLayers) + 2    # Todas las ocultas mas la de input y la de output
        prev_layer = inputLayer

        #Inicializacion de las thetas
        self.thetas = []
        for layer in self.hiddenLayers:
            self.thetas.append(np.random.uniform(low = -epsilon, high =  epsilon, size = (layer, prev_layer + 1))) #Incluimos sesgo
            prev_layer = layer

        self.thetas.append(np.random.uniform(low = -epsilon, high = epsilon, size = (outputLayer, self.hiddenLayers[-1] + 1))) #Output layer


    """
    Reset the theta matrix created in the constructor by both theta matrix manualy loaded.
    Args:
        theta1 (array_like): Weights for the first layer in the neural network.
        theta2 (array_like): Weights for the second layer in the neural network.
    """
    def new_trained(self,thetas):
        self.thetas = thetas
        

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
    Computes de sigmoid derivation of de activation (private)
    Args:
        a (array_like): activation received by the layer.
    """   
    def _sigmoidPrime(self,a):
        return a * (1 - a)


    """
    Run the feedfordward neural network step
    Args:
        x (array_like): input of the neural network.
	Return 
	------
	a (array_like): activation functions of each layers
    z (array_like): signal fuction of two last layers
    """
    def feedforward(self,x):
        a = []
        z = []
        
        a.append(x)
        for i in range(0, len(self.thetas)):
            a[i] = np.hstack([np.ones((a[i].shape[0],1)),a[i]])
            z.append(a[i] @ self.thetas[i].T)
            a.append(self._sigmoid(z[i]))

        return a, z


    """
    Computes only the cost of a previously generated output (private)
    Args:
        yPrime (array_like): output generated by neural network.
        y (array_like): output from the dataset
        lambda_ (scalar): regularization parameter
	Return 
	------
	J (scalar): the cost.
    """
    def compute_cost(self, yPrime,y, lambda_):
        epsilon = 1e-15  # Valor pequeño para evitar log(0)
        yPrime = np.clip(yPrime, epsilon, 1 - epsilon)  # Recorta los valores de yPrime

        valor = np.sum(y * np.log(yPrime) + (1 - y) * np.log(1 - yPrime)) 
        J = valor

        J *= (-1 / y.shape[0])  # Escalar el coste promedio
        J += self._regularizationL2Cost(y.shape[0], lambda_)  # Agregar la regularización
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
        return np.argmax(a3, axis = 1)
    

    """
    Compute the gradients of both theta matrix parámeters and cost J

    Args:
        x (array_like): input of the neural network.
        y (array_like): output of the neural network.
        lambda_ (scalar): regularization.

	Return 
	------
	J: cost
    grads: the gradient matrix (same shape than thetas)
    """
    def compute_gradients(self, x, y, lambda_):
        # Paso 1: Feedforward para calcular activaciones y valores intermedios
        a, z = self.feedforward(x)
        J = self.compute_cost(a[-1], y, lambda_)  # Coste total (incluye regularización)

        # Paso 2: Inicializar los errores (uno por capa)
        error = []
        error.append(np.zeros((x.shape[0], self.input)))  # Primera capa (igual que input)
        for i in range(1, self.nlayers - 1):  # Capas ocultas
            error.append(np.zeros((x.shape[0], self.hiddenLayers[i - 1])))
        error.append(np.zeros((x.shape[0], self.output)))  # Última capa (igual que output)

        # Paso 3: Calcular errores usando retropropagación
        error[-1] = a[-1] - y  # Error en la última capa
        error[-2] = np.dot(error[-1], self.thetas[-1]) * self._sigmoidPrime(a[-2])  # Penúltima capa
        for l in range(self.nlayers - 3, 0, -1):  # Capas intermedias (de penúltima a primera)
            error[l] = (np.dot(error[l + 1][:, 1:], self.thetas[l])  # Ajustar dimensiones
                        * self._sigmoidPrime(a[l]))

        # Paso 4: Calcular gradientes para cada capa de theta
        grad = [np.zeros_like(theta) for theta in self.thetas]
        grad[-1] += np.dot(error[-1].T, a[-2]) / x.shape[0]  # Última capa
        grad[-1] += self._regularizationL2Gradient(self.thetas[-1], lambda_, x.shape[0])  # Regularización

        for l in range(len(self.hiddenLayers) - 1, -1, -1):  # Desde penúltima capa hasta la primera
            grad[l] += np.dot(error[l + 1][:, 1:].T, a[l]) / x.shape[0]
            grad[l] += self._regularizationL2Gradient(self.thetas[l], lambda_, x.shape[0])  # Regularización

        return J, grad
    

    """
    Compute L2 regularization gradient
    Args:
        theta (array_like): a theta matrix to calculate the regularization.
        lambda_ (scalar): regularization.
        m (scalar): the size of the X examples.
	Return 
	------
	L2 Gradient value
    """
    def _regularizationL2Gradient(self, theta, lambda_, m):
        reg = np.zeros_like(theta)
        reg[:, 1:] = (lambda_ / m) * (theta[:,1:])
        return reg
    
    
    """
    Compute L2 regularization cost

    Args:
        lambda_ (scalar): regularization.
        m (scalar): the size of the X examples.

	Return 
	------
	L2 cost value
    """

    def _regularizationL2Cost(self, m, lambda_):
        reg = 0
        for i in range(len(self.thetas)):
            valor = np.sum(np.square(self.thetas[i][:,1:])) 
            reg += valor
        reg = lambda_ * (1 / (2*m)) * reg
        return reg
    
    
    def backpropagation(self, x, y, alpha, lambda_, numIte, verbose=0):
        Jhistory = []
        for i in range(numIte):
            J, gradients = self.compute_gradients(x, y, lambda_)
            
            # Solo añade J a Jhistory si no es NaN
            if not math.isnan(J):  # O usa np.isnan(J) si prefieres
                Jhistory.append(J)
            
            # Con los gradientes calculados actualizamos los pesos de las thetas
            for j in range(len(self.thetas)):   
                self.thetas[j] -= alpha * gradients[j]

            if verbose > 0:
                if i % verbose == 0 or i == (numIte - 1):
                    print(f"Iteration {(i + 1):6}: Cost {float(J):8.4f}")
        
        return Jhistory
    


"""
target_gradient function of gradient test 1
"""
def target_gradient(input_layer_size,hidden_layer_size,num_labels,x,y,reg_param):
    mlp = MLP(input_layer_size, [hidden_layer_size], num_labels)
    J, gradients = mlp.compute_gradients(x, y, reg_param)
    return J, gradients[0], gradients[1], mlp.thetas[0], mlp.thetas[1]


"""
costNN function of gradient test 1
"""
def costNN(Theta1, Theta2,x, ys, reg_param):
    mlp = MLP(x.shape[1], [1], ys.shape[1])
    mlp.new_trained(Theta1,Theta2)
    J, gradients = mlp.compute_gradients(x, ys, reg_param)
    return J, gradients[0], gradients[1]


"""
mlp_backprop_predict 2 to be execute test 2
"""
def MLP_backprop_predict(X_train,y_train, X_test, hiddenLayers,alpha, lambda_, num_ite, verbose):
    mlp = MLP(X_train.shape[1], hiddenLayers ,y_train.shape[1])
    Jhistory = mlp.backpropagation(X_train, y_train, alpha, lambda_, num_ite, verbose)
    a = mlp.feedforward(X_test)[0]
    a3= a[-1]
    y_pred = mlp.predict(a3)
    return y_pred