# import os
# print(os.getcwd())

import numpy as np
from tinygrad.tensor import Tensor
import tinygrad.optim as optim
from tinygrad.utils import fetch_mnist

from tqdm import trange

X_train, Y_train, X_test, Y_test = fetch_mnist()

# train a model
def layer_init(m, h):
    ret = np.random.uniform(-1., 1., size=(m,h))/np.sqrt(m*h)
    return ret.astype(np.float32)

# model
class TinyNet:
    def __init__(self):
       self.l1 = Tensor(layer_init(784, 128))
       self.l2 = Tensor(layer_init(128, 10))

    def forward(self, x):
        return x.dot(self.l1).relu().dot(self.l2).logsoftmax()


model = TinyNet()
optim = optim.SGD([model.l1, model.l2], lr=0.01)
batch_size = 128
lr=0.01

losses, accuracies = [], []

loop = trange(1000)
for i in loop:
    samp = np.random.randint(0, X_train.shape[0], size=(batch_size))
    x = Tensor(X_train[samp].reshape((-1, 28*28)))
    Y = Y_train[samp]
    y = np.zeros((len(samp), 10), np.float32)
    y[range(y.shape[0]), Y] = -1.0
    y = Tensor(y)

    outs = model.forward(x)

    loss = outs.mul(y).mean()
    loss.backward()
    optim.step() # SGD

    cat = np.argmax(outs.data, axis=1)
    accuracy = (cat == Y).mean()

    loss = loss.data
    losses.append(loss)
    accuracies.append(accuracy)
    loop.set_description("loss %.2f accurcy %.2f" % (loss, accuracy))


# evaluate
def numpy_eval():
    Y_test_preds_out = model.forward(Tensor(X_test.reshape((-1, 28*28))))
    Y_test_preds = np.argmax(Y_test_preds_out.data, axis=1)
    return (Y_test == Y_test_preds).mean()
  

accuracy = numpy_eval()
print("test accuracy is %f" % accuracy)
assert accuracy > 0.95
