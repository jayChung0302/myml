# 함수 편하게 만들기
# 1. python 함수로 이용하기

''' 이전 모습
x = Variable(np.array(0.5))
f = square() # 함수 인스턴스 생성하고
y = f(x) # 인스턴스 호출 -> 번거롭다
'''
import numpy as np
import sys
from utils import * # utils 에 업데이트된 class 들 모아둠

sys.path.append('/Users/chung/workspace/myml/deep-scratch/steps/utils.py')

def square(x):
    f = Square()
    return f(x)

def exp(x):
    f = Exp()
    return f(x)

class Variable:
        def __init__(self, data):
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

if __name__ == '__main__':
    # x = Variable(np.array(0.5))
    # a = square(x)
    # b = exp(a)
    # y = square(b)
    # y.grad = np.array(1.0)
    # y.backward()
    # print(x.grad)

    # x = Variable(np.array(0.5))
    # y = square(exp(square(x)))
    # y.grad = np.array(1.0)
    # y.backward()
    # print(x.grad)
    
    # type check 추가 후
    x = Variable(np.array(1.0)) # ok
    x = Variable(None) # ok
    # x = Variable(1.0) # NG!

    # numpy 의 독특한 관례
    x = np.array([1.0]) # 1차원 ndarray. 문제안됨
    y = x ** 2
    print(type(x), x.ndim)
    print(type(y))

    x = np.array(1.0) # 0차원 ndarray. 문제됨
    y = x ** 2
    print(type(x), x.ndim) # numpy.ndarray
    print(type(y)) # numpy.float64

    def as_array(x):
        if np.isscalar(x):
            return np.array(x)
        return x
