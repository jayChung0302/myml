# Implementing backpropagation automation
# 핵심은 만들어진 경위를 저장해놓는 것
import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
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

if __name__ == '__main__':
    A = Square()
    B = Exp()
    C = Square()
    
    x = Variable(np.array(0.5))
    a = A(x)
    b = B(a)
    y = C(b)

    # 계산 그래프의 노드들을 거꾸로 거슬러 올라가기
    # assert 문은 조건문이 True 가 아니면 예외 발생하게함.
    assert y.creator == C
    assert y.creator.input == b
    assert y.creator.input.creator == B
    assert y.creator.input.creator.input == a
    assert y.creator.input.creator.input.creator == A
    assert y.creator.input.creator.input.creator.input == x
    
    # 연결이 실제 계산을 수행하는 시점에 만들어진다.
    # 즉, 순전파로 데이터를 흘려보낼 때(Run) 생기게 되므로 Define-by-Run 이라고 함
    # 위와같은 구조가 LinkedList 임
    
    # 역전파! b->C->y
    y.grad = np.array(1.0)

    C = y.creator # 1. 함수를 가져온다.
    b = C.input # 2. 함수의 입력을 가져온다.
    b.grad = C.backward(y.grad) # 함수의 backward 메서드를 호출한다.

    # a->B->b
    B = b.creator
    a = B.input
    a.grad = B.backward(b.grad)

    # x->A->a
    A = a.creator
    x = A.input
    x.grad = A.backward(a.grad)
    print(x.grad)
    
# backward 메서드 추가

    class Variable:
        def __init__(self, data):
            self.data = data
            self.grad = None
            self.creator = None
        
        def set_creator(self, func):
            self.creator = func
        
        def backward(self):
            f = self.creator # 1. 함수를 가져온다.
            if f is not None:
                x = f.input # 2. 함수의 입력을 가져온다.
                x.grad = f.backward(self.grad) # 3. 함수의 backward 메서드를 호출한다.
                x.backward() # 하나 앞 변수의 backward 메서드를 호출한다. (재귀)

    A = Square()
    B = Exp()
    C = Square()

    x = Variable(np.array(0.5))
    a = A(x)
    b = B(a)
    y = C(b)

    # backprop
    y.grad = np.array(1.0)
    y.backward()
    print(x.grad)
