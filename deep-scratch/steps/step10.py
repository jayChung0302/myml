# python test
import unittest
import numpy as np
from utils import Variable, square

class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.0)
        self.assertEqual(y.data, expected)

    def test_backward(self):
        x = Variable(np.array(3.0))
        y = square(x)
        y.backward()
        expected = np.array(6.0)
        self.assertEqual(x.grad, expected)
    
    def test_gradient_check(self):
        x = Variable(np.random.rand(1)) # 무작위 입력값 생성
        y = square(x)
        y.backward()
        num_grad = numerical_diff(square, x)
        flg: bool = np.allclose(x.grad, num_grad) # 두 값이 거의 비슷한지 확인
        self.assertTrue(flg)

def numerical_diff(f, x, eps=1e-4):
    # 수치미분
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    output = (y1.data - y0.data) / (2 * eps)
    return output

unittest.main()
