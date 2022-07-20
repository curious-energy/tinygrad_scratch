from functools import partialmethod
import numpy as np
from tinygrad.utils import im2col, col2im
# try:
#     from numba import jit
# except ImportError:
#     jit = lambda x: x

class Tensor:
    def __init__(self, data):
        #print(type(data), data)
        if type(data) != np.ndarray:
            print("error constructing tensor with %r" % data)
            assert(False)
        if data.dtype == np.float64:
            # print("Are you sure you want float64 in %r" % data)
            pass
        self.data = data
        self.grad = None

        # internal variables used for autograd graph construction
        self._ctx = None

    def __repr__(self):
        return "Tensor %r with grad %r" % (self.data, self.grad)

    @property
    def shape(self):
        return self.data.shape

    @staticmethod
    def zeros(*shape):
        return Tensor(np.zeros(shape, dtype=np.float32))
    
    @staticmethod
    def randn(*shape):
        return Tensor(np.random.randn(*shape).astype(np.float32))

    def backward(self, allow_fill=True):
        if self._ctx is None:
            return

        if self.grad is None and allow_fill:
            # fill in the frist grad with one
            assert self.data.size == 1
            self.grad = np.ones_like(self.data)

        assert(self.grad is not None)

        grads = self._ctx.backward(self._ctx, self.grad)
        if len(self._ctx.parents) == 1:
            grads = [grads]
        for t, g in zip(self._ctx.parents, grads):
            if g is None:
                continue
            if g.shape != t.data.shape:
                print("grad shape must match tensor shape in %r, %r != %r" %
                      (self._ctx, g.shape, t.data.shape))
                assert(False)
            t.grad = g
            t.backward(False)

    def mean(self):
        div = Tensor(np.array([1/self.data.size], dtype=self.data.dtype))
        return self.sum().mul(div)


class Function:
    def __init__(self, *tensors):
        self.parents = tensors
        self.saved_tensors = []

    def save_for_backward(self, *x):
        self.saved_tensors.extend(x)

    # note that due to how partialmethod works, self and arg are switched
    def apply(self, arg, *x):
        # support the args in both orders
        if type(arg) == Tensor:
            op = self
            x = [arg]+list(x)
        else:
            op = arg
            x = [self]+list(x)
        ctx = op(*x)
        ret = Tensor(op.forward(ctx, *[t.data for t in x]))
        ret._ctx = ctx
        return ret

def register(name, fxn):
    setattr(Tensor, name, partialmethod(fxn.apply, fxn))


class Mul(Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return x * y

    @staticmethod
    def backward(ctx, grad_out):
        x, y = ctx.saved_tensors
        return y*grad_out, x*grad_out
register('mul', Mul)


class Add(Function):
    @staticmethod
    def forward(ctx, x, y):
        return x+y

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, grad_output
register('add', Add)


class ReLU(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return np.maximum(input, 0)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.copy()
        grad_input[input < 0] = 0
        return grad_input
register('relu', ReLU)


class Dot(Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)
        return input.dot(weight)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        grad_input = grad_output.dot(weight.T)
        grad_weight = grad_output.T.dot(input).T
        return grad_input, grad_weight
register('dot', Dot)


class Sum(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return np.array([input.sum()])

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output * np.ones_like(input)
register('sum', Sum)


class LogSoftmax(Function):
    @staticmethod
    def forward(ctx, input):
        def logsumexp(x):
            c = x.max(axis=1)
            return c + np.log(np.exp(x-c.reshape((-1, 1))).sum(axis=1))
        output = input - logsumexp(input).reshape((-1, 1))
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        return grad_output - np.exp(output)*grad_output.sum(axis=1).reshape((-1, 1))
register('logsoftmax', LogSoftmax)


class Reshape(Function):
    @staticmethod
    def forward(ctx, x, shape):
        ctx.save_for_backward(x.shape)
        return x.reshape(shape)

    @staticmethod
    def backward(ctx, grad_output):
        in_shape, = ctx.saved_tensors
        return grad_output.reshape(in_shape), None
register("reshape", Reshape)


class Conv2D(Function):
    @staticmethod
    def forward(ctx, x, w):
        ctx.save_for_backward(x, w)
        cout, cin, H, W = w.shape
        ret = np.zeros((x.shape[0], cout, x.shape[2]-(H-1), x.shape[3]-(W-1)), dtype=w.dtype)
        tw = w.reshape(w.shape[0], -1).T
        for Y in range(ret.shape[2]):
            for X in range(ret.shape[3]):
                tx = x[:, :, Y:Y+H, X:X+W].reshape(x.shape[0], -1)
                ret[:, :, Y, X] = tx.dot(tw)
        return ret

    @staticmethod
    def backward(ctx, grad_output):
        x, w = ctx.saved_tensors
        cout, cin, H, W = w.shape
        dx, dw = np.zeros_like(x), np.zeros_like(w)
        tw = w.reshape(w.shape[0], -1)
        for Y in range(grad_output.shape[2]):
            for X in range(grad_output.shape[3]):
                gg = grad_output[:, :, Y, X]
                tx = x[:, :, Y:Y+H, X:X+W].reshape(x.shape[0], -1)
                dx[:, :, Y:Y+H, X:X+W] += gg.dot(tw).reshape(dx.shape[0], dx.shape[1], H, W)
                dw += gg.T.dot(tx).reshape(dw.shape)
        return dx, dw
# register('conv2d', Conv2D)


# fast about 0.2s in single pass
class FastConv2D(Function):
    @staticmethod
    def forward(ctx, x, w):
        cout, cin, H, W = w.shape
        tw = w.reshape(cout, -1).T
        bs, oy, ox = x.shape[0], x.shape[2]-(H-1), x.shape[3]-(W-1)

        tx = im2col(x, H, W)

        ctx.save_for_backward(tx, w)
        ret = tx.dot(tw).reshape(bs, oy, ox, cout)
        return np.moveaxis(ret, [0, 1, 2, 3], [0, 2, 3, 1])

    @staticmethod
    def backward(ctx, grad_output):
        bs, _, oy, ox = grad_output.shape
        tx, w = ctx.saved_tensors
        cout, cin, H, W = w.shape
        tw = w.reshape(w.shape[0], -1)

        ggt = np.moveaxis(grad_output, [0, 1, 2, 3], [1, 0, 2, 3]).reshape(cout, -1)
        dw = ggt.dot(tx).reshape(w.shape)

        dxi = ggt.T.dot(tw)
        dx = col2im(dxi, H, W, oy+(H-1), ox+(W-1))
        return dx, dw
register('conv2d', FastConv2D)


class MaxPool2x2(Function):
    @staticmethod
    def forward(ctx, x):
        stack = []
        for Y in range(2):
            for X in range(2):
                stack.append(x[:,:,Y::2,X::2][None])
        stack = np.concatenate(stack, axis=0)
        idxs = np.argmax(stack, axis=0)
        ctx.save_for_backward(idxs)
        return np.max(stack, axis=0)

    @staticmethod
    def backward(ctx, grad_output):
        idxs, = ctx.saved_tensors
        s = grad_output.shape
        ret = np.zeros((s[0], s[1], s[2]*2, s[3]*2), dtype=grad_output.dtype)
        for Y in range(2):
            for X in range(2):
                ret[:,:,Y::2,X::2] = grad_output * (idxs == (Y*2+X))
        return ret
register('maxpool2x2', MaxPool2x2)