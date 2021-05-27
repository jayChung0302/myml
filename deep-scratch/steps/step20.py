# 연산자 오버로드

import numpy as np
from utils import *

class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y
    
    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy * x1, gy * x0

def mul(x0, x1):
    f = Mul()
    return f(x0, x1)

class Variable:
    def __init__(self, data, name=None): # name 지정
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f'{type(data)} type is not available.')
        self.data = data
        self.name = name
        self.creator = None
        self.grad = None
        self.generation = 0
    
    def __len__(self):
        return len(self.data)
    
    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return 'variable(' + p +')'
    
    def __mul__(self, other):
        return mul(self, other)
    
    def __add__(self, other):
        return add(self, other)

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def backward(self, retain_grad=False):
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x:x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            # gys = [output.grad for output in f.outputs]
            gys = [output().grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)
            
            for x, gx in zip(f.inputs, gxs):
                # 조건문으로 grad 가 이미 있을 시엔 더해주도록 변경
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx
                
                if x.creator is not None:
                    # funcs.append(x.creator)
                    add_func(x.creator)
            if not retain_grad:
                for y in f.outputs:
                    y().grad = None # y 는 약한 참조(weakref)

    def cleargrad(self):
        self.grad = None
    
    @property
    def shape(self):
        return self.data.shape
    
    @property
    def ndim(self):
        return self.data.ndim
    
    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

if __name__ == '__main__':
    a = Variable(np.array(3.0))
    b = Variable(np.array(2.0))
    c = Variable(np.array(1.0))
    
    # y = add(mul(a, b), c) # 귀찮
    y = a * b + c # 연산자 오버로드 후
    
    y.backward()

    print(y)
    print(a.grad)
    print(b.grad)
