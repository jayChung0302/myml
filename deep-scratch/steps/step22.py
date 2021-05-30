from utils import *

class Neg(Function):
    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy

class Sub(Function):
    def forward(self, x0, x1):
        y = x0 - x1
        return y
    
    def backward(self, gy):
        return gy, -gy

class Div(Function):
    def forward(self, x0, x1):
        y = x0 / x1
        return y
    
    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)
        return gx0, gx1

class Pow(Function):
    def __init__(self, c):
        self.c = c
    
    def forward(self, x):
        y = x ** self.c
        return y
    
    def backward(self, gy):
        x = self.inputs[0].data
        c = self.c
        gx = c * x ** (c - 1) * gy
        return gx

def pow(x, c):
    f = Pow(c)
    return f(x)

def div(x0, x1):
    x1 = as_array(x1)
    f = Div()
    return f(x0, x1)

def rdiv(x0, x1):
    x1 = as_array(x1)
    f = Div()
    return f(x1, x0)

def neg(x):
    f = Neg()
    return f(x)

def sub(x0, x1):
    x1 = as_array(x1)
    f = Sub()
    return f(x0, x1)

def rsub(x0, x1):
    x1 = as_array(x1)
    f = Sub()
    return f(x1, x0)

Variable.__neg__ = neg
Variable.__rsub__ = rsub

x = Variable(np.array(2.0))
y1 = 2.0 - x
y2 = x - 1.0
print(y)
