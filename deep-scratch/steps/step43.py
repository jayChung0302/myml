if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import dezero.functions as F
from dezero.core import *
from dezero.utils import *

import matplotlib.pyplot as plt


# generate nonlinear data samples
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)
plt.scatter(x, y, label='nonlinear data samples')

# parameter initialization
I, H, O = 1, 10, 1
W1 = Variable(0.01 * np.random.randn(I, H))
b1 = Variable(np.zeros(H))
W2 = Variable(0.01 * np.random.randn(H, O))
b2 = Variable(np.zeros(O))

# sigmoid function
def sigmoid_simple(x):
    x = as_variable(x)
    y = 1 / (1 + dezero.core.exp(-x))
    return y

def predict(x):
    y = F.linear_simple(x, W1, b1)
    y = F.sigmoid_simple(y)
    y = F.linear_simple(y, W2, b2)
    return y

lr = 0.2
iters = 10000

# NN training
for i in range(iters):
    y_pred = predict(x)
    loss = F.mean_squared_error(y, y_pred)
    
    W1.cleargrad()
    b1.cleargrad()
    W2.cleargrad()
    b2.cleargrad()
    loss.backward()

    W1.data -= lr * W1.grad.data
    b1.data -= lr * b1.grad.data
    W2.data -= lr * W2.grad.data
    b2.data -= lr * b2.grad.data
    if i % 1000 == 0:
        print(loss)

x = np.arange(0, 1, 0.001)
x = np.expand_dims(x, axis=1)
y = predict(x)
plt.plot(x.data, y.data, label='trained NN')
plt.legend()
plt.show()
