import numpy as np
from typing import ClassVar

def as_array(x):
        if np.isscalar(x):
            return np.array(x)
        return x

# 재귀에서 반복문으로
class Variable:
        def __init__(self, data: np.ndarray):
            if data is not None:
                if not isinstance(data, np.ndarray):
                    raise TypeError(f'{type(data)} type is not available.')
            self.data = data
            self.grad = None
            self.creator = None
        
        def set_creator(self, func):
            self.creator = func
        
        def backward(self):
            if self.grad is None:
                self.grad = np.ones_like(self.data)
            funcs = [self.creator]
            while funcs:
                f = funcs.pop()  # 1. 함수를 가져온다.
                x, y = f.input, f.output # 2. 함수의 입력을 가져온다.
                x.grad = f.backward(y.grad) # 3. backward 메서드를 호출한다.

                if x.creator is not None:
                    funcs.append(x.creator) # 4. 하나 앞의 함수(들)를 리스트에 추가한다.
            # 재귀가 비효율적인 이유는 호출시마다 중간결과를 메모리 스택에 쌓으면서 처리를 이어가기 때문

            
class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(as_array(y))
        output.set_creator(self) # 출력 변수에 creator 를 설정한다.
        self.input = input
        self.output = output # output 도 저장한다.
        return output
    
    def forward(self):
        raise NotImplementedError
    
    def backward(self):
        raise NotImplementedError

# 연결된 Variable 과 Function 이 있으면 backprop 자동화 가능
# 동적 계산 그래프(Dynamic computational graph) 는 실제 계산이 이뤄질 때 변수에 관련 '연결'을 기록하는 방식으로 만들어짐.
# -> PyTorch, Chainer 도 비슷한 방식

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

def square(x):
    f = Square()
    return f(x)

def exp(x):
    f = Exp()
    return f(x)
