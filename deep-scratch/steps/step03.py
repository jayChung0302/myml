import numpy as np
from step02 import *

class Exp(Function):
    def forward(self, x):
        return np.exp(x)

if __name__ == '__main__':
    x = Variable(np.array(16))
    f = Square()
    y = f(x)
    print(y.data)

    # calculate y = (e^(x^2))^2 on x = 0.5
    A = Square()
    B = Exp()
    C = Square()
    x = Variable(np.array(0.5))
    a = A(x)
    b = B(a)
    y = C(b)
    print(y.data)
    

    