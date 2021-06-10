if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import math
import numpy as np
import matplotlib.pyplot as plt
import dezero
from dezero import optimizers
import dezero.functions as F
import dezero.datasets as datasets
from dezero.models import MLP

x, t = datasets.get_spiral(train=True)
print(x.shape)
print(t.shape)

print(x[10], t[10])
print(x[110], t[110])

max_epoch = 300
batch_size = 30
hidden_size = 10
lr = 1.0

net = MLP((hidden_size, 3))
optimizer = optimizers.SGD(lr).setup(net)

data_size = len(x)
print(data_size)
max_iter = math.ceil(data_size/batch_size)

for epoch in range(max_epoch):
    index = np.random.permutation(data_size)
    sum_loss = 0

    for i in range(max_iter):
        batch_index = index[i * batch_size : (i+1) * batch_size]
        batch_x = x[batch_index]
        batch_t = t[batch_index]

        y = net(batch_x)
        loss = F.softmax_cross_entropy_simple(y, batch_t)
        net.cleargrads()
        loss.backward()
        optimizer.update()
        sum_loss += float(loss.data) * len(batch_t)
    
    avg_loss = sum_loss/data_size
    print(f'epoch: {epoch + 1}, loss: {avg_loss}')
