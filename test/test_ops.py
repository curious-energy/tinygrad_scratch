import unittest
import torch
import numpy as np
from tinygrad.tensor import Tensor

import timeit
import functools

def test_op(shapes, torch_fxn, tinygrad_fxn, atol=1e-7, grad_atol=1e-7):
    ts = [torch.rand(x, requires_grad=True) for x in shapes]
    tst = [Tensor(x.detach().numpy()) for x in ts]

    out = torch_fxn(*ts)
    ret = tinygrad_fxn(*tst)

    np.testing.assert_allclose(ret.data, out.detach().numpy(), atol=atol)

    out.mean().backward()
    ret.mean().backward()

    for t, tt in zip(ts, tst):
        np.testing.assert_allclose(t.grad, tt.grad, atol=grad_atol)

    # speed
    torch_fp = timeit.Timer( )
    torch_fp = timeit.Timer(functools.partial(torch_fxn, *ts)).timeit(5) * 1000/5
    tinygrad_fp = timeit.Timer(functools.partial(tinygrad_fxn, *tst)).timeit(5) * 1000/5

    torch_fbp = timeit.Timer(functools.partial(lambda f,x: f(*x).mean().backward(), torch_fxn, ts)).timeit(5) * 1000/5
    tinygrad_fbp = timeit.Timer(functools.partial(lambda f,x: f(*x).mean().backward(), tinygrad_fxn, tst)).timeit(5) * 1000/5

    print("testing %30r   torch/tinygrad fp: %.2f / %.2f ms  bp: %.2f / %.2f ms" % (shapes, torch_fp, tinygrad_fp, torch_fbp-torch_fp, tinygrad_fbp-tinygrad_fp))


class TestOps(unittest.TestCase):
    def test_conv2d(self):
        for bs in [1, 128]:
            for cin in [1, 3]:
                for H in [2, 5]:
                    for W in [2, 3, 5]:
                        test_op([(bs, cin, 11, 8), (4, cin, H, W)], lambda x, w: torch.nn.functional.conv2d(x, w).relu(), lambda x, w: Tensor.conv2d(x, w).relu(), atol=2e-5, grad_atol=2e-6)

    def test_maxpool2d(self):
        test_op([(32, 2, 110, 28)], lambda x: torch.nn.functional.max_pool2d(x, (2, 2)), Tensor.max_pool2d)

    def test_avgpool2d(self):
        test_op([(32, 2, 111, 28)], lambda x: torch.nn.functional.avg_pool2d(x, (2, 2)), Tensor.avg_pool2d)
        


if __name__ == '__main__':
    unittest.main(verbosity=2)
