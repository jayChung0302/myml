# backpropagation. 역전파
# numerical diff 의 문제: 1. 계산량이 많다. 
# 변수가 여러개인 계산을 미분할 경우 변수 각각을 미분해야 하기 때문
# 2. 정확도 문제. dx 값 때문에 오차가 발생.
# 역전파를 이용하면 효율적인 미분계산 가능, 오차도 적다.

# 역전파에서 전파되는 데이터는 모두 y 에 대한 미분값임을 알 수 있음.
# 손실함수의 각 매개변수에 대한 미분을 계산
# 역전파 시 미분 값에 순전파에서 계산된 값이 필요하게 됨 (메모리에 저장)
# weight term 의 backprop 에서 activation 이 들어감
# bias term 의 backprop 시엔 activation term 이 필요하진 않음 

import numpy as np
