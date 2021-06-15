if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import time
import dezero
from dezero import optimizers
from dezero.dataloaders import DataLoader
from dezero.models import MLP
import dezero.functions as F
import dezero.datasets as datasets

max_epoch = 3
batch_size = 100

train_set = dezero.datasets.MNIST(train=True)
train_loader = DataLoader(train_set, batch_size)
model = MLP((1000, 10))
optimizer = optimizers.SGD().setup(model)

# 매개변수 읽기
if os.path.exists('my_mlp.npz'):
    model.load_weights('my_mlp.npz')

if dezero.cuda.gpu_enable:
    train_loader.to_gpu()
    model.to_gpu()
for epoch in range(max_epoch):
    start = time.time()
    sum_loss = 0

    for x, t in train_loader:
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()
        sum_loss += float(loss.data) * len(t)
    
    elapsed_time = time.time() - start
    print(f'epoch: {epoch+1}, loss: {sum_loss/len(train_set):.4f}, time: {elapsed_time:.4f}')
    
model.save_weights('my_mlp.npz')
