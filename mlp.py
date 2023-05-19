import numpy as np

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

class MLP:
    def __init__(self, sizes: list[int], batchSize: int, f: str = 'tanh'):
        self.S = sizes
        self.L = len(self.S)
        self.N = batchSize
        self.Y = [np.empty(shape=(self.N, self.S[i - 1] + 1)) for i in range(self.L)]
        self.W = [None] + [np.random.normal(0, 0.5, (self.S[i - 1] + 1, self.S[i])) for i in range(1, self.L)]
        self._f = f

    
    def _add_bias(self, V):
        bias = np.ones((len(V),1))
        return np.hstack([V, bias])

    def _sub_bias(self, V):
        return V[:, :-1]
    
    @property
    def f(self):
        return self._f

    @f.setter
    def f(self, fun):
        if fun != 'tanh' and fun != 'sigmoid': 
            raise Exception('Activation function should be  \'tanh\' or \'sigmoid\'')
    
    def activation(self, Xh):
        self.Y = [np.empty(shape=(self.N, self.S[i - 1] + 1)) for i in range(self.L)]
        Y_b = Xh
        for k in range(1, self.L):
            self.Y[k - 1] = self._add_bias(Y_b)
            if self.f == 'sigmoid':
                Y_b = sigmoid(self.Y[k - 1] @ self.W[k])
            else:
                Y_b = np.tanh(self.Y[k - 1] @ self.W[k])
        self.Y[self.L - 1] = Y_b
        return self.Y
    
    def correction(self, Yh, Zh):
        D = [np.empty(shape=(self.N, self.S[i - 1] + 1)) for i in range(self.L)]
        deltaW = [None] + [np.empty(shape=(self.S[i - 1] + 1, self.S[i])) for i in range(1, self.L)]
        E = Zh - Yh[self.L - 1]
        if self.f == 'sigmoid':
            dY = Yh[self.L - 1] * (1 - Yh[self.L - 1]) # Derivative of sigmoid
        else:
            dY = 1 - np.square(Yh[self.L - 1]) # Derivative of tanh
        D[self.L - 1] = E * dY
        for k in reversed(range(1, self.L)):
            deltaW[k] = Yh[k - 1].T @ D[k]
            E = D[k] @ self.W[k].T
            if self.f == 'sigmoid':
                dY = Yh[k - 1] * (1 - Yh[k - 1]) # Derivative of sigmoid
            else:
                dY = 1 - np.squared(Yh[k - 1]) # Derivative of tanh
            D[k - 1] = self._sub_bias(E * dY)
        return deltaW

    def predict(self, x):
      return self.activation(x)[-1]