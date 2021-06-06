if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import dezero.functions as F
from dezero.core import as_variable, Variable, Function

if __name__ == '__main__':
    x = Variable(np.array([1, 2, 3, 4, 5, 6]))
    y = F.sum(x)
    y.backward()
    print(y)
    print(x.grad)

    # 2x3 matrix
    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    y = F.sum(x)
    y.backward()
    print(y)
    print(x.grad)

    x = np.array([[1,2,3],[4,5,6]])
    y = np.sum(x, axis=0)
    z = np.sum(x, axis=1)
    print(y, z)
    print(x.shape, '->', y.shape, z.shape)

    x = np.array([[1,2,3],[4,5,6]])
    y = F.sum(x, axis=0)
    y.backward()
    print(y)
    print(x.grad)

    x = Variable(np.random.randn(2,3,4,5))
    y = x.sum(keepdims=True)
    print(y.shape)
