# 지금까지 구현은 pop() 만 되어있어 우선순위가 정해지지 않음
# 단순히 DFS 식으로 구현되어있음
# forward 시 세대를 추가해 backward 가 잘 작동하게 하려고 함.
# 1. 인스턴스 변수 generation 을 추가
# 2. generation 의 순서로 꺼내기
import numpy as np

class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f'{type(data)} type is not available.')
        self.data = data
        self.creator = None
        self.grad = None
        self.generation = 0 # 세대 수를 기록하는 변수 추가
    
    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1 # 세대 기록. 함수의 부모세대 + 1

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        # funcs
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
            gys = [output.grad for output in f.outputs]
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

    def cleargrad(self):
        self.grad = None

class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]
        
        self.generation = max([x.generation for x in inputs]) # 
        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self):
        raise NotImplementedError
    
    def backward(self):
        raise NotImplementedError

class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y
    
    def backward(self, gy):
        return gy, gy

def add(x0, x1):
    f = Add()
    return f(x0, x1)

def as_array(x):
        if np.isscalar(x):
            return np.array(x)
        return x

if __name__ == '__main__':
    # dummy 실험
    generations = [2, 0, 1, 4, 2]
    funcs = []
    for g in generations:
        f = Function() # 더미 함수 클래스
        f.generation = g
        funcs.append(f)
    
    funcs.sort(key=lambda x: x.generation)
    print([f.generation for f in funcs])
    f = funcs.pop()
    print(f.generation)
