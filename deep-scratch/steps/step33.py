if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable
import matplotlib.pyplot as plt

# y = x^4 - 2x^2 optimization with newton's method
def f(x):
    y = x**4 - 2 * x**2
    return y

if __name__ == '__main__':
    iters=10
    # x = Variable(np.array(2.0))
    # y = f(x)
    # y.backward(create_graph=True)
    # print(x.grad)

    # gx = x.grad
    # x.cleargrad() # 미분값 재설정
    # gx.backward()
    # print(x.grad)

    iters=10
    x = Variable(np.array(2.0))
    for i in range(iters):
        print(i, x)
        y = f(x)
        x.cleargrad()
        y.backward(create_graph=True)

        gx = x.grad
        x.cleargrad()
        gx.backward()
        gx2 = x.grad
        x.data = x.data - gx.data/gx2.data
        
