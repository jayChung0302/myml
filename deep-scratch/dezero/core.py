# -*- coding: utf-8 -*-
import numpy as np
import dezero
import numpy as np
from typing import ClassVar
import weakref
import contextlib

def as_array(x, array_module=np):
    if np.isscalar(x):
        return array_module.array(x)
    return x

def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)
class Variable:
    __array_priority__ = 200
    def __init__(self, data, name=None):
        try:
            import cupy
            array_types = (np.ndarray, cupy.ndarray)
        except ImportError:
            array_types = (np.ndarray)
        if data is not None:
            if not isinstance(data, array_types):
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

    def __neg__(self, other):
        return neg(self, other)
    
    def __sub__(self, other):
        return sub(self, other)
        
    def __rsub__(self, other):
        return rsub(self, other)
    
    def __truediv__(self, other):
        return div(self, other)
    
    def __rtruediv__(self, other):
        return rdiv(self, other)

    def __pow__(self, other):
        return pow(self, other)

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1
    
    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return dezero.functions.reshape(self, shape)

    def transpose(self):
        return dezero.functions.transpose(self)
    
    def sum(self, axis=None, keepdims=False):
        return dezero.functions.sum(self, axis, keepdims)

    def backward(self, retain_grad=False, create_graph=False):
        if self.grad is None:
            xp = dezero.cuda.get_array_module(self.data)
            self.grad = Variable(xp.ones_like(self.data))
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
            gys = [output().grad for output in f.outputs]
            with using_config('enable_backprop', create_graph):
                gxs = f.backward(*gys)
                if not isinstance(gxs, tuple):
                    gxs = (gxs,)
                
                for x, gx in zip(f.inputs, gxs):
                    # ??????????????? grad ??? ?????? ?????? ?????? ??????????????? ??????
                    if x.grad is None:
                        x.grad = gx
                    else:
                        x.grad = x.grad + gx
                    
                    if x.creator is not None:
                        # funcs.append(x.creator)
                        add_func(x.creator)
                if not retain_grad:
                    for y in f.outputs:
                        y().grad = None # y ??? ?????? ??????(weakref)

    def cleargrad(self):
        self.grad = None
    
    def to_cpu(self):
        if self.data is not None:
            self.data = dezero.cuda.as_numpy(self.data)
    
    def to_gpu(self):
        if self.data is not None:
            self.data = dezero.cuda.as_cupy(self.data)
    
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
    
    @property
    def T(self):
        return dezero.functions.transpose(self)

class Config:
    # ?????? ???????????? ??? ??? ???????????? ??????????????? ??????. ????????? ???????????? ??????????????? ?????? ?????? ????????? ?????? ????????? ??????
    enable_backprop = True
    train = True

class Function:
    def __call__(self, *inputs):
        inputs = [as_variable(x) for x in inputs] # variable ???
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]
        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs]) # ?????? ??????
            for output in outputs:
                output.set_creator(self) # ?????? ??????
            self.inputs = inputs
            # self.outputs = outputs
            self.outputs = [weakref.ref(output) for output in outputs] # ??????????????? ??????

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self):
        raise NotImplementedError
    
    def backward(self):
        raise NotImplementedError

# ????????? Variable ??? Function ??? ????????? backprop ????????? ??????
# ?????? ?????? ?????????(Dynamic computational graph) ??? ?????? ????????? ????????? ??? ????????? ?????? '??????'??? ???????????? ???????????? ????????????.
# -> PyTorch, Chainer ??? ????????? ??????

@contextlib.contextmanager
def using_config(name: str, value: bool):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)

def test_mode():
    return using_config('train', False)
    
def no_grad():
        return using_config('enable_backprop', False)

class Add(Function): # Function ?????? unpack ??? tuple ?????? ?????? ???????????? ???????????? ?????????
    def forward(self, x0, x1): # ????????? ????????? ??? ??? ??????.
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 + x1
        return y
    
    def backward(self, gy):
        gx0, gx1 = gy, gy
        # forward ?????? broadcast ??????????????? shape ??? ???????????? backward ??? ?????? ?????? broadcast ??? ????????? ??????
        if self.x0_shape != self.x1_shape:
            gx0 = dezero.functions.sum_to(gx0, self.x0_shape)
            gx1 = dezero.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1 # ????????? ???????????? 1????????? ?????? ??????????????? gradient ??? ????????? ????????????.

class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y
    
    def backward(self, gy):
        x0, x1 = self.inputs
        return gy * x1, gy * x0

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

class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y
    
    def backward(self, gy):
        x = self.inputs[0]
        gx = 2 * x * gy
        return gx

class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y
    
    def backward(self, gy):
        y = self.inputs[0]
        gx = gy * y
        return gx

class Log(Function):
    def forward(self, x):
        y = np.log(x)
        return y
    
    def backward(self, gy):
        gx = gy * (1 / self.inputs[0])
        return gx

class Div(Function):
    def forward(self, x0, x1):
        y = x0 / x1
        return y
    
    def backward(self, gy):
        x0, x1 = self.inputs
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
        x, = self.inputs
        c = self.c
        gx = c * x ** (c - 1) * gy
        return gx

class Parameter(Variable):
    pass

def add(x0, x1):
    x1 = as_array(x1, dezero.cuda.get_array_module(x0.data))
    f = Add()
    return f(x0, x1)

def mul(x0, x1):
    x1 = as_array(x1, dezero.cuda.get_array_module(x0.data))
    f = Mul()
    return f(x0, x1)

def square(x):
    f = Square()
    return f(x)

def exp(x):
    f = Exp()
    return f(x)

def log(x):
    f = Log()
    return f(x)

def neg(x):
    f = Neg()
    return f(x)
    
def sub(x0, x1):
    x1 = as_array(x1, dezero.cuda.get_array_module(x0.data))
    f = Sub()
    return f(x0, x1)

def rsub(x0, x1):
    x1 = as_array(x1, dezero.cuda.get_array_module(x0.data))
    f = Sub()
    return f(x1, x0)

def div(x0, x1):
    x1 = as_array(x1, dezero.cuda.get_array_module(x0.data))
    f = Div()
    return f(x0, x1)

def rdiv(x0, x1):
    x1 = as_array(x1, dezero.cuda.get_array_module(x0.data))
    f = Div()
    return f(x1, x0)

def pow(x, c):
    f = Pow(c)
    return f(x)

def setup_variable():
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
    Variable.__neg__ = neg
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__pow__ = pow
    Variable.__getitem__ = dezero.functions.get_item
