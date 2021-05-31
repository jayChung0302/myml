# sine function
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable, Function

class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y
    
    def backward(self, gy):
        x = self.inputs[0].data
        gx =  gy * np.cos(x)
        return gx
    
def sin(x):
    f = Sin()
    return f(x)

def mysin(x, threshold=0.0001):
    import math
    y = 0
    for i in range(100000):
        c = (-1) ** i / math.factorial(2 * i + 1)
        t = c * x ** (2 * i + 1)
        y = y + t
        if abs(t.data) < threshold:
            break
    return y

if __name__ == '__main__':
    x = Variable(np.array(np.pi/4))
    y = sin(x)
    y.backward()
    print(y.data)
    print(x.grad)
    
    x = Variable(np.array(np.pi/4))
    y = mysin(x)
    print(y.data)
