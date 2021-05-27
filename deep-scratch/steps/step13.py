# Add 함수 backpropagation 구현
# Variable 의 backward 메서드가 여러개 출력이 가능하도록 수정
# Square 함수 구현
import numpy as np

def as_array(x):
        if np.isscalar(x):
            return np.array(x)
        return x

class Function:
    def __call__(self, *inputs): # * 붙인다. 리스트를 사용하는 대신 임의 개수의 인수를 건네 함수 호출 가능.
        xs = [x.data for x in inputs]
        ys = self.forward(*xs) # 별표를 붙여 unpack
        if not isinstance(ys, tuple): # tuple 인스턴스가 아닐 경우 추가 지원
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]
        
        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs
        return outputs if len(outputs) > 1 else outputs[0]
        # 리스트의 원소가 하나라면 첫 번째 원소를 반환 (list 벗기기)
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
    
    def set_creator(self, func):
        self.creator = func

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        
        funcs = [self.creator]
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
                    funcs.append(x.creator)

class Add(Function): # Function 에서 unpack 및 tuple 아닐 경우 튜플화로 구현했기 때문에
    def forward(self, x0, x1): # 이렇게 간결히 짤 수 있음.
        y = x0 + x1
        return y
    
    def backward(self, gy):
        return gy, gy # 덧셈의 미분값은 1이므로 상위 노드에서의 gradient 가 그대로 전달된다.

def add(x0, x1):
    f = Add()
    return f(x0, x1)

class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y
    
    def backward(self, gy):
        x = self.inputs[0].data # 수정 전: x = self.input.data
        gx = 2 * x * gy
        return gx

def square(x):
    f = Square()
    return f(x)

if __name__ == '__main__':
    x0 = Variable(np.array(2))
    x1 = Variable(np.array(3))
    y = add(x0, x1)
    print(y.data)

    # z = x^2 + y^2 계산
    x = Variable(np.array(2.0))
    y = Variable(np.array(3.0))

    z = add(square(x), square(y))
    z.backward()
    print(z.data)
    print(x.grad)
    print(y.grad)
