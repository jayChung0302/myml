if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import dezero.functions as F
from dezero.core import as_variable, Variable, Function


class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = x.reshape(self.shape)
        return y
    
    def backward(self, gy):
        return reshape(gy, self.x_shape)
        
def reshape(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)

if __name__ == '__main__':
    x = Variable(np.array([[1,2,3],[4,5,6]]))
    y = F.reshape(x, (6,))
    y.backward(retain_grad=True)
    print(x.grad)
    
    x = np.random.rand(1,2,3)

    y = x.reshape((2, 3))
    print(y.shape)
    y = x.reshape([2, 3])
    print(y.shape)
    y = x.reshape(2, 3)
    print(y.shape)

    x = Variable(np.random.randn(1, 2, 3))
    y = x.reshape((2, 3))
    y = x.reshape(2, 3)

    x = np.array([[1, 2, 3], [4, 5, 6]])
    print(x)
    y = np.transpose(x)
    print(y)

    x = Variable(x)
    y = F.transpose(x)
    y.backward()
    print(x.grad)
    
    x = Variable(np.random.rand(2, 3))
    y = x.transpose()
    print(y)
    y = y.T
    print(y)
