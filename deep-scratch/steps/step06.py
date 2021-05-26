# backprop 을 위해 미분값을 가지도록 Variable 클래스에 grad 인스턴스 변수 추가
import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None

class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        # 입력변수 보관
        self.input = input
        return output

    def forward(self):
        raise NotImplementedError
    
    def backward(self):
        raise NotImplementedError

class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y

    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx

class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y
    
    def backward(self, gy):
        x = self.input.data
        gx = gy * np.exp(x)
        return gx


if __name__ == '__main__':
    A = Square()
    B = Exp()
    C = Square()
    
    # forward pass
    x = Variable(np.array(0.5))
    a = A(x)
    b = B(a)
    y = C(b)
    print(y.data)

    # backward pass
    y.grad = np.array(1)
    b.grad = C.backward(y.grad)
    a.grad = B.backward(b.grad)
    x.grad = A.backward(a.grad)
    print(x.grad)
