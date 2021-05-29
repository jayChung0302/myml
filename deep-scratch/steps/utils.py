import numpy as np
from typing import ClassVar
import weakref
import contextlib

def as_array(x):
        if np.isscalar(x):
            return np.array(x)
        return x

def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)

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
    
    def __add__(self, other):
        return add(self, other)
    
    def __radd__(self, other):
        return add(self, other)

    def __mul__(self, other):
        return mul(self, other)
    
    def __rmul__(self, other):
        return mul(self, other)

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
    

            
class Config:
    # 설정 데이터는 단 한 군데에만 존재하는게 좋음. 따라서 클래스를 인스턴스화 하지 않고 클래스 상태 그대로 이용
    enable_backprop = True

class Function:
    def __call__(self, *inputs):
        inputs = [as_variable(x) for x in inputs] # variable 화
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]
        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs]) # 세대 설정
            for output in outputs:
                output.set_creator(self) # 연결 설정
            self.inputs = inputs
            # self.outputs = outputs
            self.outputs = [weakref.ref(output) for output in outputs] # 약한참조로 해결

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self):
        raise NotImplementedError
    
    def backward(self):
        raise NotImplementedError

# 연결된 Variable 과 Function 이 있으면 backprop 자동화 가능
# 동적 계산 그래프(Dynamic computational graph) 는 실제 계산이 이뤄질 때 변수에 관련 '연결'을 기록하는 방식으로 만들어짐.
# -> PyTorch, Chainer 도 비슷한 방식

@contextlib.contextmanager
def using_config(name: str, value: bool):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)

def no_grad():
        return using_config('enable_backprop', False)
class Add(Function): # Function 에서 unpack 및 tuple 아닐 경우 튜플화로 구현했기 때문에
    def forward(self, x0, x1): # 이렇게 간결히 짤 수 있음.
        y = x0 + x1
        return y
    
    def backward(self, gy):
        return gy, gy # 덧셈의 미분값은 1이므로 상위 노드에서의 gradient 가 그대로 전달된다.

class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y
    
    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy * x1, gy * x0

class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y
    
    def backward(self, gy):
        x = self.inputs[0].data # 수정 전: x = self.input.data
        gx = 2 * x * gy
        return gx

class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y
    
    def backward(self, gy):
        x = self.inputs[0].data
        gx = gy * np.exp(x)
        return gx

def add(x0, x1):
    x1 = as_array(x1)
    f = Add()
    return f(x0, x1)

def mul(x0, x1):
    x1 = as_array(x1)
    f = Mul()
    return f(x0, x1)

def square(x):
    f = Square()
    return f(x)

def exp(x):
    f = Exp()
    return f(x)

# Variable.__add__ = add
# Variable.__radd__ = add
# Variable.__mul__ = mul
# Variable.__rmul__ = mul

if __name__ == '__main__':
    # step21.py test
    x = Variable(np.array(2.0))
    y = x + np.array(3.0)
    print(y)

    y = x + 3.0
    print(y)

    x = Variable(np.array(2.0))
    y = 3.0 * x + 1.0
    print(y)
    
