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
import numpy as np
from dezero import test_mode

x = np.ones(5)
print(x)

# if train
y = F.dropout(x)
print(y)

# if test
with test_mode():
    y = F.dropout(x)
    print(y)
