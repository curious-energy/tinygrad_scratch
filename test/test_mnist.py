import os
import unittest
import numpy as np
from tinygrad.ops import Tensor
import tinygrad.optim as optim
from tinygrad.utils import layer_init_uniform, fetch_mnist

from tqdm import trange

X_train, Y_train, X_test, Y_test = fetch_mnist()


# model
class TinyNet:
    def __init__(self):
       self.l1 = Tensor(layer_init_uniform(784, 128))
       self.l2 = Tensor(layer_init_uniform(128, 10))

    def forward(self, x):
        return x.dot(self.l1).relu().dot(self.l2).logsoftmax()

class TinyConvNet:
    def __init__(self):
        chans = 16
        conv = 7
        self.c1 = Tensor(layer_init_uniform(chans, 1, conv, conv))
        self.l1 = Tensor(layer_init_uniform(((28-conv+1)**2)*chans, 128))
        self.l2 = Tensor(layer_init_uniform(128, 10))

    def forward(self, x):
        x.data = x.data.reshape((-1, 1, 28, 28))
        x = x.conv2d(self.c1).relu()
        x = x.reshape(Tensor(np.array((x.shape[0], -1))))
        return x.dot(self.l1).relu().dot(self.l2).logsoftmax()

# train a model
def train(model, optim, steps, batch_size = 128):
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
        optim.step()

        cat = np.argmax(out.data, axis=1)
        accuracy = (cat == Y).mean()

        loss = loss.data
        losses.append(loss)
        accuracies.append(accuracy)
        loop.set_description("loss %.2f accurcy %.2f" % (loss, accuracy))


def evaluate(model):
    def numpy_eval():
        Y_test_preds_out = model.forward(Tensor(X_test.reshape((-1, 28*28)).astype(np.float32)))
        Y_test_preds = np.argmax(Y_test_preds_out.data, axis=1)
        return (Y_test == Y_test_preds).mean()

    accuracy = numpy_eval()
    print("test accuracy is %f" % accuracy)
    assert accuracy > 0.95
    

class TestMNIST(unittest.TestCase):
    def test_conv(self):
        np.random.seed(1337)
        model = TinyConvNet()
        optimizer = optim.Adam([model.c1, model.l1, model.l2], lr=0.001)
        train(model, optimizer, steps=400)
        evaluate(model)

    def test_sgd(self):
        np.random.seed(1337)
        model = TinyNet()
        optimizer = optim.SGD([model.l1, model.l2], lr=0.001)
        train(model, optimizer, steps=1000)
        evaluate(model)

    def test_rmsprop(self):
        np.random.seed(1337)
        model = TinyNet()
        optimizer = optim.RMSprop([model.l1, model.l2], lr=0.0002)
        train(model, optimizer, steps=1000)
        evaluate(model)

if __name__ == '__main__':
    unittest.main()