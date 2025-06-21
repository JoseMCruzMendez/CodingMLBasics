import numpy as np
from numpy import ndarray
from typing import Union, Callable
import math, numbers
from functools import wraps
from Optimizers.GradGraph import Variable #I had to refactor Variable, since calling super would return variables and not tensors
#Instead of creating Variable(res), I had to edit the method to return self.__class()__(res). This seems to be called "covariant return issue"? The fix seems present in numpy

def ensure_ndarray(f: Callable):
    #I've never used wraps before, getting it to work was a bit puzzling
    @wraps(f)
    def type_check(self, other: Union["Tensor", "Variable", ndarray, float, int], *args, **kwargs):
        if isinstance(other, Tensor):
            v = other
        elif isinstance(other, Variable):
            v = Tensor(other.v)
            v.grad = other.grad
            v._prev = other._prev
            v._backward = other._backward
        elif isinstance(other, ndarray):
            v =  Tensor(np.atleast_2d(other))
        elif isinstance(other, numbers.Real):
            v =  Tensor(np.atleast_2d(np.array(other, dtype=np.float32)))
        else:
            raise TypeError(f"Unsupported type {type(other)}")
        bound = f.__get__(self, type(self))          # re-bind to keep method semantics
        return bound(v, *args, **kwargs)
    return type_check


def unbroadcast(g, target_shape):
    """
    Sum the gradient `g` over the axes that were broadcast
    so that the result has `target_shape`.
    """
    #I'm not sure if I would've been able to come up with this myself, I am unfamiliar with broadcasting logic
    # --- scalar target: just sum everything and return 0-D array -----------
    if target_shape == ():                      # or len(target_shape) == 0
        return np.asarray(g.sum())              # shape ()
    # 1. Pad target_shape on the left so ndim matches
    while len(target_shape) < g.ndim:
        target_shape = (1,) + target_shape

    # 2. Any axis where target size == 1 (or target had no axis)
    #    must be summed out.
    axes = tuple(
        i for i, (g_dim, t_dim) in enumerate(zip(g.shape, target_shape))
        if t_dim == 1
    )
    if axes:                        # no-op when axes == ()
        g = g.sum(axis=axes, keepdims=True)

    # 3. Finally reshape to exactly target_shape
    return g.reshape(target_shape)

class Tensor(Variable):
    def __init__(self, v: ndarray):
        v = np.asarray(v)
        super().__init__(v)
        self.grad = np.zeros_like(v)

    def _accumulate_grad(self, g):
        g = unbroadcast(g, self.v.shape)
        super()._accumulate_grad(g)

    @ensure_ndarray
    def __add__(self, other: Union["Variable", ndarray, float, int]):
        return super().__add__(other)

    __radd__ = __add__

    @ensure_ndarray
    def __mul__(self, other):
        return super().__mul__(other)

    __rmul__ = __mul__

    @ensure_ndarray
    def __truediv__(self, other):
        return super().__truediv__(other)

    @ensure_ndarray
    def __rtruediv__(self, other):
        return other/self

    def pow_both_tensors(self, other):#Had to reimplement since I call math.log
        other = self._new(other) if not isinstance(other, Variable) else other
        out = self._new(self.v ** other.v)
        out._prev = [self, other]

        def _backward():
            # base gradient – always safe
            self._accumulate_grad(other.v * (self.v ** (other.v - 1)) * out.grad)
            # exponent gradient – only if base ≠ 0
            safe_vals = self.v.copy()
            signs = np.sign(self.v)
            safe_vals[np.isclose(self.v, 0)] = 1e-6
            other._accumulate_grad(out.v * signs * np.log(np.abs(safe_vals)) * out.grad)
        out._backward = _backward
        return out


    def __pow__(self, other):
        #This is a case of bad single responsibility, I could've refactored the original Variable but I want to leave this here as an example. #Ended up refactoring due to _accumulate_gradient refactor
        if isinstance(other, numbers.Real) or isinstance(other, ndarray):
            super().pow_single_tensor(other)
        return self.pow_both_tensors(other)

    @ensure_ndarray
    def __neg__(self):
        out = Tensor(-self.v)
        out._prev = [self]
        def _backward():
            self.grad += -1 * out.grad
        out._backward = _backward
        return out


    def update_v(self, updater: Callable[[ndarray, ndarray], ndarray]):
        self.grad[~np.isfinite(self.grad)] = 0.0
        grad = np.clip(self.grad, -5, 5)#clips gradients to prevent explosions
        self.v = updater(self.v, grad)

    def zero_grad(self):
        self.grad = np.zeros_like(self.v)
        self._prev = []


    #Now to actually start implementing new methods
    def sum(self, axis=None, keepdims=False):
        #broadcasting to axis and using keepdims was not something I thought of in foresight. I am having a bit of a hard time
        #Wrapping my head around how they are supposed to work for sum, mean and matmul despite the fact numpy handles most of the work
        res = np.sum(self.v, axis=axis, keepdims=keepdims)
        out = self._new(res)
        out._prev = [self]
        shape = self.v.shape
        def _backward():
            self.grad += np.broadcast_to(out.grad, shape)
        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        res = np.mean(self.v, axis=axis, keepdims=keepdims)
        out = self._new(res)
        out._prev = [self]
        shape = self.v.shape
        #Takes either the full shape of the array or just the relevant axis we averaged
        elems_averaged = np.prod(self.v.shape if axis is None else np.take(self.v.shape, axis))
        def _backward():
            self.grad += np.ones_like(self.v) * np.broadcast_to(out.grad, shape) / elems_averaged
        out._backward = _backward
        return out

    @ensure_ndarray
    def matmul(self, other: "Tensor"):
        A, B = np.atleast_2d(self.v), np.atleast_2d(other.v)
        res = np.matmul(A, B)
        out = self._new(res)
        out._prev = [self, other]
        def _backward():
            grad = np.atleast_2d(out.grad)
            #This gradient was more complicated than I thought, but I think I understand the trace/permutation/inner product logic
            self.grad += np.matmul(grad, B.swapaxes(-1, -2)) #swapping dims for batch dim
            other.grad += np.matmul(A.swapaxes(-1, -2), grad)
        out._backward = _backward
        return out

    def exp(self): #need for softmax
        res = np.exp(self.v)
        out = self._new(res)
        out._prev = [self]
        def _backward():
            self.grad += out.grad * res
        out._backward = _backward
        return out

    #Had to implement for convnets to prevent weird behavior in gradgraph
    def flatten(self):
        res = self.v.flatten()
        out = self._new(res)
        out._prev = [self]
        def _backward():
            self.grad += np.reshape(out.grad, self.v.shape)
        out._backward = _backward
        return out

#Helper functions necessary for NN, at first they were in the class but I think they make more sense outside

def relu(var: Tensor):
    res = np.maximum(var.v, 0)
    out = Tensor(res)
    out._prev = [var]
    def _backward():
        var.grad += (res > 0) * out.grad
    out._backward = _backward
    return out

def leaky_relu(var: Tensor, negative_slope=0.01):
    res = np.maximum(var.v * negative_slope, var.v)
    out = Tensor(res)
    out._prev = [var]
    def _backward():
        var.grad += (var.v > 0) * out.grad + (var.v < 0) * out.grad * negative_slope
    out._backward = _backward
    return out

def sigmoid(var: Tensor)->Tensor:
    safe_var = np.clip(var.v, -50, 50) #Had an overflow error that kept occuring
    res = 1/(1 + np.exp(-safe_var))
    out = Tensor(res)
    out._prev = [var]
    def _backward():
        var.grad += out.grad * res * (1 - res)
    out._backward = _backward
    return out

def BCELoss(probs: Tensor, target: Union[Tensor, ndarray], reduction="mean"):
    target = target.v if isinstance(target, Tensor) else target
    res = -target * np.log(probs.v) - (1. - target) * np.log(1 - probs.v)
    out = Tensor(res)
    out._prev = [probs]
    def _backward():
        eps = 1e-12
        p = np.clip(probs.v, eps, 1 - eps)
        grad_update = (p - target)/(p * (1 - p))
        probs.grad += out.grad * grad_update #Originally This was *= grad_update instead of what it is now, which led to a hard to track bug
    out._backward = _backward
    if reduction == "mean":
        return out.mean()
    elif reduction == "sum":
        return out.sum()
    else:
        raise ValueError("Invalid reduction mode, can be mean or sum")

def softmax(z:Tensor):
    z_shift = z - z.v.max(axis=-1, keepdims=True)   # stability
    exps = z_shift.exp()
    return exps / exps.sum(axis=-1, keepdims=True)

def logloss(probs: Tensor, target: Union[Tensor, ndarray], reduction="mean"):
    target = target.v if isinstance(target, Tensor) else target
    probs = softmax(probs)
    p = np.clip(probs.v, 1e-12, 1 - 1e-12)
    res = -(target * np.log(p) + (1-target) * np.log(1-p)).sum(axis=-1, keepdims=True)
    out = Tensor(res)
    out._prev = [probs]
    if reduction == "mean":
        out = out.mean()
    elif reduction == "sum":
        out = out.sum()
    else:
        raise ValueError("Invalid reduction mode, can be mean or sum")

    def _backward():
        grad_logits = (p - target) * out.grad
        probs._accumulate_grad(grad_logits)
    out._backward = _backward
    return out

def rmse(preds: Tensor, true: Tensor, reduction="mean"):
    diff_sq = (preds - true)**2.
    mse = diff_sq.mean() if reduction == "mean" else diff_sq.sum()
    out = mse ** 0.5
    def _backward():
        n = diff_sq.v.size
        grad_rmse = out.grad / (2 * out.v + 1e-8)
        preds._accumulate_grad(2*(preds.v - true) / n * grad_rmse)
    out._backward = _backward
    return out



class GradModule:

    def __init__(self):
        self.params: list[Tensor] = []

    @ensure_ndarray
    def forward(self, value: Tensor, *args, **kwargs):
        raise NotImplementedError()

    def __call__(self, value, *args, **kwargs):
        return self.forward(value,*args, **kwargs)

    def get_params(self):
        return self.params

    def update(self, updater: Callable[[ndarray, ndarray], ndarray]):
        """Interface necessary for optimizers to update parameters"""
        for param in self.params:
            param.update_v(updater)

    def zero_grad(self):
        """Interface necessary for optimizers to zero gradients"""
        for param in self.params:
            param.zero_grad()