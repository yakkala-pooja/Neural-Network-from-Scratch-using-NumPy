import numpy as np

np.random.seed(42)

# Class Layer with forward and backward propagation functions which aren't implemented and __init__ function is acting as a placeholder
class Layer:
    def __init__(self):
        pass
    
    def forward_propagation(self, inputData):
        raise NotImplementedError
    
    def backward_propagation(self, outputerr, lr):
        raise NotImplementedError

# The class FCLayer initializes weights and biases randomly using He Initialization for stability in the performance
class FCLayer(Layer):
    def __init__(self, inputSize, outputSize):
        self.weights = np.random.randn(inputSize, outputSize) * np.sqrt(2 / inputSize)
        self.biases = np.zeros((1, outputSize))

    # In the forward propagation I used the formula X . H + B to get the output
    def forward_propagation(self, inputData):
        self.input = inputData
        self.output = np.dot(inputData, self.weights) + self.biases
        return self.output

    # For the backward propagation, I found the gradient with respect to weights and bias and the input error and
    def backward_propagation(self, outputerr, lr):
        dw = np.dot(self.input.T, outputerr)
        db = np.sum(outputerr, axis=0, keepdims=True)
        inputerr = np.dot(outputerr, self.weights.T)

        # Then I updated the weights and bias using the formula w = w - lr * dw and same with bias 
        self.weights = self.weights - lr * dw
        self.biases = self.biases - lr * db
        
        # Here I returned the input error so that it can continue to the next back propagation step
        return inputerr

# The class activation function has all the required functions. First it starts with initializing the activation and derivative of activation
class ActivationLayer(Layer):
    def __init__(self, activation, dactivation):
        self.activation = activation
        self.dactivation = dactivation

    # Then in forward propagation, I just use the activation functio to get the output
    def forward_propagation(self, inputData):
        self.input = inputData
        self.output = self.activation(inputData)
        return self.output

    # In the backward propagation, I use the formula output error (Gradient) * derivative of the activation function(X)
    def backward_propagation(self, outputerr, lr):
        return outputerr * self.dactivation(self.input)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# This is the derivative of the sigmoid
def sigderivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def relu(x):
    return np.maximum(0, x)

# This is the derivative of the ReLU
def reluderivative(x):
    return np.where(x > 0, 1, 0)

def leakyrelu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

# This is the derivative of the Leaky ReLU
def LRderivative(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)

def softmax(x):
    expx = np.exp(x - np.max(x, axis=1, keepdims=True))
    return expx / np.sum(expx, axis=1, keepdims=True)

# This is the derivative of the softmax
def smderivative(x):
    s = softmax(x)
    return s * (1 - s)

def tanh(x):
    return np.tanh(x)

# This is the derivative of the tanh
def tanhderivative(x):
    return 1 - np.tanh(x) ** 2

def mse(ytrue, ypred):
    return np.mean(np.square(ytrue - ypred))

# This is the derivative of the Mean Squared Error
def msederivative(ytrue, ypred):
    ypred = np.clip(ypred, 1e-9, 1 - 1e-9)
    return 2 * (ypred - ytrue) / ytrue.shape[0]

def binaryce(ytrue, ypred):
    ypred = np.clip(ypred, 1e-9, 1 - 1e-9)
    return -np.mean(ytrue * np.log(ypred) + (1 - ytrue) * np.log(1 - ypred))

# This is the derivative of the Binary Cross Entropy
def bceDerivative(ytrue, ypred):
    ypred = np.clip(ypred, 1e-9, 1 - 1e-9)
    return -(ytrue / ypred - (1 - ytrue) / (1 - ypred))

def ccentropy(ytrue, ypred):
    ypred = np.clip(ypred, 1e-9, 1 - 1e-9)
    return -np.sum(ytrue * np.log(ypred)) / ytrue.shape[0]

# This is the derivative of the Categorical Cross Entropy
def ccederivative(ytrue, ypred):
    ypred = np.clip(ypred, 1e-9, 1 - 1e-9)
    return -ytrue / ypred

# The class Network has the following required functions add, predict, with additional function for getting the loss function, and fit function
class Network:
    # Here I have initialized the list for layers and the loss function as well as the derivative of the loss function (dl)
    def __init__(self):
        self.layers = []
        self.loss = None
        self.dl = None

    # the add function works as intended in the question (it added the layers specified by the users)
    def add(self, layer):
        self.layers.append(layer)

    # the predict function takes input data and simply runs the forward_propagation function to get the output
    def predict(self, inputData):
        result = inputData
        for layer in self.layers:
            result = layer.forward_propagation(result)
        return result

    # This is an additional function that takes the last layer of the network into account and selects the appropriate loss function
    def getlossfun(self, actName):
        if actName == "sigmoid":
            return binaryce, bceDerivative
        elif actName == "softmax":
            return ccentropy, ccederivative
        else:
            return mse, msederivative

    # The fit function here takes in both training and validation data and returns the losses of both
    def fit(self, X, Y, Xval, Yval, epochs, lr, lossfun=None):
        # First if the loss function is not specified (lossfun) and if the last layer is an activation layer then I call the getlossfun function to automatically select a loss function, else I use the lossfun specified by the user
        lastLayer = self.layers[-1]
        if isinstance(lastLayer, ActivationLayer) and lossfun is None:
            actName = lastLayer.activation.__name__
            self.loss, self.dl = self.getlossfun(actName)
        else:
            self.loss, self.dl = lossfun

        # Then I create the losses for both train and test and run my epoches
        losses = []
        valLosses = []
        for epoch in range(epochs):
            output = X
            for layer in self.layers:
                output = layer.forward_propagation(output)
        # I use forward propagation for output and find the loss for the data
            lossval = self.loss(Y, output)
            losses.append(lossval)
            # Then I compute the error and start the back propagation
            error = self.dl(Y, output)
            for layer in reversed(self.layers):
                error = layer.backward_propagation(error, lr)
            # I start with the validation set and find the forward propagation for each point and check the loss
            val_output = Xval
            for layer in self.layers:
                val_output = layer.forward_propagation(val_output)
            valLoss = self.loss(Yval, val_output)
            valLosses.append(valLoss)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {lossval:.6f}, Validation Loss: {valLoss:.6f}")
        return losses, valLosses