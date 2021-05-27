# 모든 변수가 미분값을 변수에 저장해두는 비효율 개선 -> Variable
# Config 클래스 만들어서 Function에서 참조,
# enable_backprop 을 통해 역전파 활성모드 설정
# contextlib.contextmanager 데코레이션을 통해 with 문을 사용한 모드 전환 가능.

import numpy as np
import weakref
from utils import *

class Config:
    # 설정 데이터는 단 한 군데에만 존재하는게 좋음. 따라서 클래스를 인스턴스화 하지 않고 클래스 상태 그대로 이용
    enable_backprop = True

class Function:
    def __call__(self, *inputs):
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

class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f'{type(data)} type is not available.')
        self.data = data
        self.creator = None
        self.grad = None
        self.generation = 0
    
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

if __name__ == '__main__':
    x0 = Variable(np.array(1.0))
    x1 = Variable(np.array(1.0))
    t = add(x0, x1)
    y = add(x0, t)
    y.backward()

    print(y.grad, t.grad)
    print(x0.grad, x1.grad)  

    # 모드전환
    Config.enable_backprop = True
    x = Variable(np.ones((100,100,100)))
    y = square(square(square(x)))
    y.backward()
    print(x.grad)

    Config.enable_backprop = False
    x = Variable(np.ones((100,100,100)))
    y = square(square(square(x)))
    print(x.grad)

    # with using_config('enable_backprop', False):
    #     x = Variable(np.array(2.0))
    #     y = square(x)
    
    # contextlib.contextmanager 데코레이션을 통해 with 문을 사용한 모드 전환 가능.
    # contextlib 활용 예제
    import contextlib
    
    @contextlib.contextmanager
    def config_test():
        print('start') # 전처리
        try:
            yield
        finally:
            print('done') # 후처리
    
    with config_test():
        print('process...')

    @contextlib.contextmanager
    def using_config(name: str, value: bool):
        old_value = getattr(Config, name)
        setattr(Config, name, value)
        try:
            yield
        finally:
            setattr(Config, name, old_value)
    
    with using_config('enable_backprop', False):
        x = Variable(np.array(2.0))
        y = square(x)

    # 간단하게.
    def no_grad():
        return using_config('enable_backprop', False)
    
    with no_grad():
        x = Variable(np.array(2.0))
        y = square(x)
