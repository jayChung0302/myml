import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data

class Function:
    def __call__(self, input: Variable):
        x = input.data # 데이터를 꺼낸다
        y = x ** 2 # 실제 계산.
        output = Variable(y)
        return output


if __name__ == '__main__':
    # Variable 예제
    data = np.array(1.0)
    x = Variable(data)
    print(x.data)
    x.data = np.array(2.0)
    print(x.data)
    # 0차원 배열은 스칼라, 1차원 배열은 벡터
    # 2차원 배열은 행렬
    # 배열은 텐서라고도 부를 수 있다.

    # Function 예제
    x = Variable(np.array(10))
    f = Square()
    y = f(x)
    print(type(y)) # type() 함수는 객체의 클래스를 알려준다.
    print(y.data)
