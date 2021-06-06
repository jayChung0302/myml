# 선형 회귀
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from numpy.core.fromnumeric import var
import dezero.functions as F
from dezero.core import as_variable, Variable, Function
from dezero.utils import *

import matplotlib.pyplot as plt

# toy dataset
np.random.seed(0)
x = np.random.rand(100, 1)
y = 5 + 2*x + np.random.rand(100, 1) # add random noise to y
W = Variable(np.zeros((1, 1)))
b = Variable(np.zeros(1))

plt.scatter(x, y)

def predict(x):
    y = F.matmul(x, W) + b
    return y

def mean_squared_error(x0, x1):
    diff = x0 - x1
    return F.sum(diff ** 2) / len(diff)
    
lr = 0.1
iters = 100
for i in range(iters):
    y_pred = predict(x)
    loss = mean_squared_error(y, y_pred)

    W.cleargrad()
    b.cleargrad()
    loss.backward()

    W.data = W.data - lr * W.grad.data
    b.data = b.data - lr * b.grad.data
    print(W, b, loss)

x = np.arange(0,1, 0.001)
y = 2*x + 5
plt.plot(x, y, label='Ideal')
x = np.arange(0,1, 0.001)
y = W.data * x + b
print(x.size, y.size)
plt.plot(x, y.data[0], label='Linear Regression')
plt.legend()
plt.show()



if __name__ == '__main__':
    pass
