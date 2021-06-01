if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable
import matplotlib.pyplot as plt

def rosenbrock(x0, x1):
    y = 100 * (x1 - x0**2)**2 + (1 - x0)**2
    return y

if __name__ == '__main__':
    # gradient Descent
    x0 = Variable(np.array(0.0))
    x1 = Variable(np.array(2.0))
    lr = 0.001 # learning rate
    iters = 50000
    x0_data = []
    x1_data = []
    for i in range(iters):
        y = rosenbrock(x0, x1)

        x0.cleargrad()
        x1.cleargrad()

        y.backward()

        x0.data = x0.data - x0.grad * lr
        x1.data = x1.data - x1.grad * lr
        x0_data.append(x0.data)
        x1_data.append(x1.data)

    plt.scatter(x0_data, x1_data)
    plt.show()

