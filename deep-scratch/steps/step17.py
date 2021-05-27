# 기존 코드: 효율성 좀 배제함. 메모리 효율 증가시켜보자
# python 의 메모리 관리 방식 1. 참조 카운트(참조 reference 를 세는 방식)
# - 모든 객체는 참조 카운트가 0인 상태로 생성됨. 이후 다른 객체가 참조할 때마다 1씩 증가
# - 반대로 객체에 대한 참조가 끊길 때마다 1씩 감소하다가 0이 되면 파이썬 인터프리터가 회수.
# - ex) 대입연산자 사용, 함수에 인수로 전달, 컨테이너 타입 객체(리스트, 튜플, 클래스 등)에 추가할 때 참조 카운트 증가
# dezero 의 순환참조 문제 해결 ㄱㄱ
import numpy as np

class obj:
    pass

def f(x):
    print(x)

if __name__ == '__main__':
    a = obj() # 변수에 대입: 참조 카운트 1
    f(a) # 함수에 전달: 함수 안에선 참조 카운트 2
    # 함수 완료: 빠져나오면 참조 카운트 1
    a = None # 대입 해제: 참조 카운트 0

    a = obj()
    b = obj()
    c = obj()
    a.b = b
    b.c = c

    a=b=c=None # 도미노처럼 참조 카운트가 0이되어 삭제됨

    # 순환참조 예.
    a = obj()
    b = obj()
    c = obj()
    a.b = b
    b.c = c
    c.a = a
    a=b=c=None # 각 객체의 참조카운트는 0이 되지 않고, 메모리에서 삭제되지 않음.
    # 그래서 GC를 통해 메모리를 관리함.
    # GC의 구조는 생략됨.
    # 명시적으로 부르거나, 메모리가 부족해지는 시점에 인터프리터에 의해 자동 호출됨.
    # 파이썬에선 weakref.ref 사용하여 약한 참조 구현 가능
    import weakref
    a = np.array([1,2,3])
    b = weakref.ref(a)
    c = a
    print(b)
    print(b()) # 참조된 데이터에 접근하기
    a = None
    print(b) # dead 로 출력됨
    print(c)
    
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

        def backward(self):
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
            # self.outputs = outputs
            self.outputs = [weakref.ref(output) for output in outputs] # 약한참조로 해결
            return outputs if len(outputs) > 1 else outputs[0]

        def forward(self):
            raise NotImplementedError
        
        def backward(self):
            raise NotImplementedError

    # 순환참조 없앤 뒤 참조구조 예제로 확인
    for i in range(10):
        x = Variable(np.random.randn(10000))
        y = square(square(square(x)))
