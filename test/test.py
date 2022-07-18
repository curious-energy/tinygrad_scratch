import numpy as np
from tinygrad.tensor import Tensor


x = Tensor(np.eye(3))
y = Tensor(np.array([[2.0, 0, -2.0]]))
z = y.dot(x).sum()
z.backward()

print(x.grad)
print(y.grad)