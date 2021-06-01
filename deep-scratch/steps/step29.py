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

def gx2(x):
    return 12 * x**2 - 4

if __name__ == '__main__':
    # gradient descent vs newton
    x = Variable(np.array(2.0))
    iters = 10
    newton_x = []
    newton_y = []
    for i in range(iters):
        if not isinstance(x.data, np.ndarray):
            newton_x.append(np.array(x.data))
        else:
            newton_x.append(x.data)
        y = f(x)
        newton_y.append(y.data)
        x.cleargrad()
        y.backward()

        x.data = x.data - x.grad / gx2(x.data)
        print(x.data, y.data)
        

    x = Variable(np.array(2.0))
    iters = 400
    lr = 0.001
    grad_descent_x = []
    grad_descent_y = []

    for i in range(iters):
        y = f(x)
        grad_descent_x.append(x.data)
        grad_descent_y.append(y.data)
        x.cleargrad()
        y.backward()

        x.data = x.data - lr * x.grad
        
    
    x = np.arange(-1.5, 2.2, 0.1)
    y = f(x)
    plt.scatter(grad_descent_x, grad_descent_y, label='Gradient Descent')
    plt.scatter(newton_x, newton_y, marker='*', color='violet', label='Newton\'s method')

    plt.plot(x, y, color='blue')
    plt.legend()
    title = r"f(x) =  $y=x^4-2x^2$"
    
    plt.title(title)
    plt.show()
