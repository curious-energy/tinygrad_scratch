import os
# print(os.getcwd())
import numpy as np
from tinygrad.tensor import Tensor
import tinygrad.optim as optim
from tinygrad.utils import layer_init_uniform,fetch_mnist

from tqdm import trange
np.random.seed(1337)

X_train, Y_train, X_test, Y_test = fetch_mnist()

# train a model
# model
class TinyNet:
    def __init__(self):
       self.l1 = Tensor(layer_init_uniform(784, 128))
       self.l2 = Tensor(layer_init_uniform(128, 10))

    def forward(self, x):
        return x.dot(self.l1).relu().dot(self.l2).logsoftmax()

class TinyConvNet:
    def __init__(self):
        self.chans = 4
        self.c1 = Tensor(layer_init_uniform(self.chans, 1, 3, 3))
        self.l1 = Tensor(layer_init_uniform(26*26*self.chans, 128))
        self.l2 = Tensor(layer_init_uniform(128, 10))

    def forward(self, x):
        x.data = x.data.reshape((-1, 1, 28, 28))
        x = x.conv2d(self.c1).reshape(Tensor(np.array((-1, 26*26*self.chans)))).relu()
        return x.dot(self.l1).relu().dot(self.l2).logsoftmax()

if os.getenv("CONV") == '1':
    model = TinyConvNet()
    optim = optim.Adam([model.c1, model.l1, model.l2], lr=0.001)
    steps = 400
else:
    model = TinyNet()
    optim = optim.SGD([model.l1, model.l2], lr=0.001)
    steps = 1000
    

batch_size = 128
losses, accuracies = [], []

loop = trange(steps)
for i in loop:
    samp = np.random.randint(0, X_train.shape[0], size=(batch_size))
    x = Tensor(X_train[samp].reshape((-1, 28*28)).astype(np.float32))
    Y = Y_train[samp]
    y = np.zeros((len(samp), 10), np.float32)
    y[range(y.shape[0]), Y] = -10.0
    y = Tensor(y)

    out = model.forward(x)

    loss = out.mul(y).mean()
    loss.backward()
    optim.step() # SGD

    cat = np.argmax(out.data, axis=1)
    accuracy = (cat == Y).mean()

    loss = loss.data
    losses.append(loss)
    accuracies.append(accuracy)
    loop.set_description("loss %.2f accurcy %.2f" % (loss, accuracy))


# evaluate
def numpy_eval():
    Y_test_preds_out = model.forward(Tensor(X_test.reshape((-1, 28*28)).astype(np.float32)))
    Y_test_preds = np.argmax(Y_test_preds_out.data, axis=1)
    return (Y_test == Y_test_preds).mean()


accuracy = numpy_eval()
print("test accuracy is %f" % accuracy)
assert accuracy > 0.95
