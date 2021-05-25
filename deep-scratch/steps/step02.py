import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data
        
class Function:
    '''
    Variable 에서 데이터 찾기, 계산하기(forward), 다시 Variable 로 포장하기
    '''
    def __call__(self, input: Variable):
        x = input.data # 데이터를 꺼낸다
        y = self.forward(x) # 호출되면 forward 함수가 적용됨. 구체적인 계산은 forward 메서드에서 한다.
        output = Variable(y)
        return output

    def forward(self, x):
        '''
        NotImplementedError 를 날려줌으로써 Function 클래스의 forward 호출한 사람에게,
        '이 메서드는 상속하여 구현해야 한다' 는 사실을 알려줄 수 있음.
        '''
        raise NotImplementedError()

class Square(Function):
    def forward(self, x):
        return x ** 2



if __name__ == '__main__':
    # Function 예제
    x = Variable(np.array(10))
    f = Square()
    y = f(x)
    print(type(y)) # type() 함수는 객체의 클래스를 알려준다.
    print(y.data)
