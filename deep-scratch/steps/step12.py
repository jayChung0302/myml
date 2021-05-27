# 귀찮은 부분 : 리스트로 다 만들어줘야 하는 점
# 인수와 결과를 직접 주고받도록 수정
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

# class Add(Function):
#     def forward(self, xs):
#         x0, x1 = xs
#         y = x0 + x1
#         return (y,)

class Add(Function): # Function 에서 unpack 및 tuple 아닐 경우 튜플화로 구현했기 때문에
    def forward(self, x0, x1): # 이렇게 간결히 짤 수 있음.
        y = x0 + x1
        return y

def add(x0, x1):
    f = Add()
    return f(x0, x1)

if __name__ == '__main__':
    # xs = [Variable(np.array(2)), Variable(np.array(3))] # 리스트로 준비 -> 귀찮음
    # f = Add()
    # ys = f(xs)
    # y = ys[0]
    # print(y.data)

    # 가변 길이 변수 예제
    def f(*x):
        print(x)
    f(1, 2, 3)
    f(1, 2 ,3 ,4, 5 ,6)

    x0 = Variable(np.array(2))
    x1 = Variable(np.array(3))
    # f = Add()
    y = add(x0, x1)
    print(y.data)
