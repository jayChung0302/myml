if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dezero import Variable, Model, as_variable
import dezero.functions as F
import dezero.layers as L
from dezero.models import MLP
from dezero import optimizers

model = MLP((10, 3))
x = np.array([[0.2, -0.4]])
y = model(x)
print(y)

def softmax1d(x):
    x = as_variable(x)
    y = F.exp(x)
    sum_y = F.sum(y)
    return y / sum_y

p = softmax1d(y)
print(p)

# Softmax 는 exp term 이 들어가기 때문에 결과값이 너무 커지거나 작아지기 쉬움.
# overflow 방지책이 필요

def softmax_simple(x, axis=1):
    x = as_variable(x)
    y = exp(x)
    sum_y = sum(y, axis=axis, keepdims=True)
    return y / sum_y
    
x = np.array([[0.2, -0.4], [0.3, 0.5], [1.3, -3.2], [2.1, 0.3]])
t = np.array([2, 0, 1, 0])
y = model(x)
loss = F.softmax_cross_entropy_simple(y, t)
print(loss)
