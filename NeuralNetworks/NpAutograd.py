import numpy as np
from numpy import ndarray
from typing import Union, Callable
import math, numbers
from functools import wraps
from Optimizers.GradGraph import Variable #I had to refactor Variable, since calling super would return variables and not tensors
#Instead of creating Variable(res), I had to edit the method to return self.__class()__(res). This seems to be called "covariant return issue"? The fix seems to be present in numpy

def ensure_ndarray(f: Callable):
    #I've never used wraps before, getting it to work was a bit puzzling
    @wraps(f)
    def type_check(self, other: Union["Variable", ndarray, float, int], *args, **kwargs):
        if isinstance(other, Variable):
            v = other
        elif isinstance(other, ndarray):
            v =  Variable(other)
        elif isinstance(other, numbers.Real):
            v =  Variable(np.array(other))
        else:
            raise TypeError(f"Unsupported type {type(other)}")
        return f(self, v, *args, **kwargs)
    return type_check

class Tensor(Variable):
    def __init__(self, v: ndarray):
        v = np.asarray(v)
        super().__init__(v)
        self.grad = np.zeros_like(v)

    @ensure_ndarray
    def __add__(self, other: Union["Variable", ndarray, float, int]):
        return super().__add__(other)

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

    @ensure_ndarray
    def pow_one_tensor(self, other):
        """Pow if only the base is a tensor"""
        out = Tensor(self.v ** other)
        out._prev = [self]
        def _backward():
            self.grad += other * (self.v ** (other - 1)) * out.grad
        out._backward = _backward
        return out

    @ensure_ndarray
    def pow_both_tensors(self, other):
        """Pow if both base and exp are tensors"""
        out = Tensor(self.v ** other.v)
        out._prev = [self, other]

        def _backward():
            # base gradient – always safe
            self.grad += other.v * (self.v ** (other.v - 1)) * out.grad
            # exponent gradient – only if base ≠ 0
            if abs(self.v) > 1e-12:
                other.grad += out.v * math.log(abs(self.v)) * out.grad
        out._backward = _backward
        return out

    def __pow__(self, other):
        #This is a case of bad single responsibility, I could've refactored the original Variable but I want to leave this here as an example.
        if isinstance(other, numbers.Real) or isinstance(other, ndarray):
            return self.pow_one_tensor(other)
        else:
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
        if not math.isfinite(self.grad):
            self.grad = np.zeros_like(self.v)
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
        out = Tensor(res)
        out._prev = [self]
        shape = self.v.shape
        def _backward():
            self.grad += np.broadcast_to(out.grad, shape)
        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        res = np.mean(self.v, axis=axis, keepdims=keepdims)
        out = Tensor(res)
        out._prev = [self]
        shape = self.v.shape
        #Takes either the full shape of the array or just the relevant axis we averaged
        elems_averaged = np.prod(self.v.shape if axis is None else np.take(self.v.shape, axis))
        def _backward():
            self.grad += np.ones_like(self.v) * np.broadcast_to(out.grad, shape) / elems_averaged
        out._backward = _backward
        return out

    def matmul(self, other):
        res = np.matmul(self.v, other.v)
        out = Tensor(res)
        out._prev = [self, other]
        def _backward():
            #This gradient was more complicated than I thought, but I think I understand the trace/permutation/inner product logic
            self.grad += np.matmul(out.grad, other.v.T)
            other.grad += np.matmul(self.v.T, out.grad)
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

def sigmoid(var: Tensor):
    res = 1/(1 + np.exp(-var.v))
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


class GradModule:

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def get_params(self):
        raise NotImplementedError()

    def update(self, updater: Callable[[ndarray, ndarray], ndarray]):
        """Interface necessary for optimizers to update parameters"""
        raise NotImplementedError()