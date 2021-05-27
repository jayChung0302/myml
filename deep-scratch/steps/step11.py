# 현재까지는 입출력 1개
# 가변 길이 입출력으로 변경
import numpy as np

def as_array(x):
        if np.isscalar(x):
            return np.array(x)
        return x

class Function:
    def __call__(self, inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(xs)
        # output = Variable(as_array(y))
        outputs = [Variable(as_array(y)) for y in ys]
        
        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs
        return outputs

    def forward(self):
        raise NotImplementedError
    
    def backward(self):
        raise NotImplementedError


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

class Add(Function):
    def forward(self, xs):
        x0, x1 = xs
        y = x0 + x1
        return (y,)

if __name__ == '__main__':
    xs = [Variable(np.array(2)), Variable(np.array(3))] # 리스트로 준비
    f = Add()
    ys = f(xs)
    y = ys[0]
    print(y.data)

