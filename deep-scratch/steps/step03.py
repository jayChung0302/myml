import numpy as np
from step02 import *

if __name__ == '__main__':
    x = Variable(np.array(16))
    f = Square()
    y = f(x)
    print(y.data)
    