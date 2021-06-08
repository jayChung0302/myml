if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dezero import Variable, Model
import dezero.functions as F
import dezero.layers as L
from dezero.models import MLP

class TwoLayerNet(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.l1 = L.Linear(hidden_size)
        self.l2 = L.Linear(out_size)
    
    def forward(self, x):
        y = F.sigmoid(self.l1(x))
        y = self.l2(y)
        return y

x = Variable(np.random.randn(5, 10), name='x')
net = TwoLayerNet(100, 10)
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

lr = 0.4
max_iter = 10000
hidden_size = 10

for i in range(max_iter):
    y_pred = net(x)
    loss = F.mean_squared_error(y, y_pred)

    net.cleargrads()
    loss.backward()

    for p in net.params():
        p.data -= lr * p.grad.data
    if i % 1000 == 0:
        print(loss)

model = MLP((10, 1))
model = MLP((10, 20, 30, 40, 1))
