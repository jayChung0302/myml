if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import dezero.functions as F
from dezero.core import as_variable, Variable, Function
from dezero.utils import *

if __name__ == '__main__':
    x = np.array([1,2,3])
    y = np.broadcast_to(x, (2,3))
    print(y)

    x = np.array([[1,2,3],[4,5,6]])
    print(x)
    y = sum_to(x, (1,3))
    print(y)

    y = sum_to(x, (2,1))
    print(y)

    # np broadcast
    x0 = np.array([1, 2, 3])
    x1 = np.array([10])
    y = x0 + x1
    print(y)

    x0 = Variable(np.array([1, 2, 3]))
    x1 = Variable(np.array([10]))
    y = x0 + x1
    print(y)

    y.backward()
    print(x1.grad)
    print(x0.grad)
    
