# import os
# print(os.getcwd())

import numpy as np
from tinygrad.tensor import Tensor
from tqdm import trange


# load the mnist dataset

def fetch(url):
  import requests, gzip, os, hashlib, numpy
  fp = os.path.join("/tmp", hashlib.md5(url.encode('utf-8')).hexdigest())
  if os.path.isfile(fp):
    with open(fp, "rb") as f:
      dat = f.read()
  else:
    with open(fp, "wb") as f:
      dat = requests.get(url).content
      f.write(dat)
  return numpy.frombuffer(gzip.decompress(dat), dtype=np.uint8).copy()
X_train = fetch("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
Y_train = fetch("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")[8:]
X_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
Y_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")[8:]


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

    cat = np.argmax(outs.data, axis=1)
    accuracy = (cat == Y).mean()

    # SGD
    model.l1.data = model.l1.data - lr * model.l1.grad
    model.l2.data = model.l2.data - lr * model.l2.grad

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
