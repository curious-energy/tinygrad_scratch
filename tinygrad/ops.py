import numpy as np
from tinygrad.utils import im2col, col2im
from tinygrad.tensor import Function, register


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
register('conv2d', Conv2D)


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
# register('conv2d', FastConv2D)

def stack_for_pool(x, py, px):
    my, mx = (x.shape[2]//py)*py, (x.shape[3]//px)*px
    stack = []
    xup = x[:, :, :my, :mx]
    for Y in range(py):
        for X in range(px):
            stack.append(xup[:, :, Y::2, X::2][None])
    return np.concatenate(stack, axis=0)


class MaxPool2D(Function):
    @staticmethod
    def forward(ctx, x):
        stack = stack_for_pool(x, 2, 2)
        idxs = np.argmax(stack, axis=0)
        ctx.save_for_backward(idxs, x.shape)
        return np.max(stack, axis=0)

    @staticmethod
    def backward(ctx, grad_output):
        idxs, s = ctx.saved_tensors
        my, mx = (s[2]//2)*2, (s[3]//2)*2
        ret = np.zeros(s, dtype=grad_output.dtype)
        for Y in range(2):
            for X in range(2):
                ret[:,:,Y:my:2,X:mx:2] = grad_output * (idxs == (Y*2+X))
        return ret
register('max_pool2d', MaxPool2D)


class AvgPool2D(Function):
    @staticmethod
    def forward(ctx, x):
        stack = stack_for_pool(x, 2, 2)
        ctx.save_for_backward(x.shape)
        return np.mean(stack, axis=0)
    
    @staticmethod
    def backward(ctx, grad_output):
        s, = ctx.saved_tensors
        my, mx = (s[2]//2)*2, (s[3]//2)*2
        ret = np.zeros(s, dtype=grad_output.dtype)
        for Y in range(2):
            for X in range(2):
                ret[:,:,Y:my:2,X:mx:2] = grad_output / 4
        return ret
register('avg_pool2d', AvgPool2D)
