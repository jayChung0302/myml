import numpy as np
from step02 import *
from step03 import *

# numerical differentiation
# The central difference is closer to the ideal difference than the forward difference.

# eps mean epsilon, represent very small value 
def numerical_diff(x, f, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data)/(2 * eps)

if __name__ == '__main__':
    # differentiate x^2 on x = 2.0
    f = Square()
    x = Variable(np.array(2.0))
    dy = numerical_diff(x, f, 1e-4)
    print(dy)

    # differentiate synthesis function
    # y = (e^(x^2))^2 on x = 0.5
    
    def s(x):
        f = Square()
        g = Exp()
        h = Square()
        return h(g(f(x)))

    x = Variable(np.array(0.5))
    print(x.data)
    dy = numerical_diff(x, s, 1e-4)
    print(dy)