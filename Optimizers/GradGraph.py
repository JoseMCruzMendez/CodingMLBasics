from typing import Union, Callable
import math, numbers

class Variable:
    def __init__(self, v):
        self.v = v
        self.grad = 0.
        self._backward = lambda: None
        self._prev = []

    def _new(self, val):
        return self.__class__(val)

    def __add__(self, other: Union["Variable", float, int]):
        other = self._new(other) if not isinstance(other, Variable) else other
        out = self._new(self.v + other.v)
        out._prev = [self, other]
        def _backward():
            self.grad += 1. * out.grad
            other.grad += 1. * out.grad
        out._backward = _backward
        return out

    __radd__ = __add__

    def __sub__(self, other):
        return self + (other * -1)

    def __mul__(self, other):
        other = self._new(other) if not isinstance(other, Variable) else other
        out = self._new(self.v * other.v)
        out._prev = [self, other]
        def _backward():
            self.grad += other.v * out.grad
            other.grad += self.v * out.grad
        out._backward = _backward
        return out

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = self._new(other) if not isinstance(other, Variable) else other
        out = self._new(self.v / other.v)
        out._prev = [self, other]
        def _backward():
            self.grad += 1/other.v * out.grad
            other.grad += (-self.v/(other.v**2)) * out.grad
        out._backward = _backward
        return out

    def __rtruediv__(self, other):
        other = self._new(other) if not isinstance(other, Variable) else other
        return other/self

    def __pow__(self, other):
        if isinstance(other, numbers.Real):
            out = self._new(self.v ** other)
            out._prev = [self]
            def _backward():
                self.grad += other * (self.v ** (other - 1)) * out.grad
            out._backward = _backward
            return out

        other = self._new(other) if not isinstance(other, Variable) else other
        out = self._new(self.v ** other.v)
        out._prev = [self, other]

        def _backward():
            # base gradient – always safe
            self.grad += other.v * (self.v ** (other.v - 1)) * out.grad
            # exponent gradient – only if base ≠ 0
            if abs(self.v) > 1e-12:
                other.grad += out.v * math.log(abs(self.v)) * out.grad
        out._backward = _backward
        return out


    def __neg__(self):
        out = self._new(-self.v)
        out._prev = [self]
        def _backward():
            self.grad += -1 * out.grad
        out._backward = _backward
        return out

    def backward(self, upstream_grad=1.0):
        topo = []
        visited = set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for p in v._prev:
                    build(p)
                topo.append(v)
        build(self)

        # Initialize gradient of the root
        self.grad = upstream_grad

        # Traverse in reverse, calling each node's backward
        for node in reversed(topo):
            node._backward()
            node._prev = []

    def update_v(self, updater: Callable[[float, float], float]):
        if not math.isfinite(self.grad):
            self.grad = 0.0
        grad = max(min(self.grad, 5), -5)#clips gradients to prevent explosions
        self.v = updater(self.v, grad)

    def zero_grad(self):
        self.grad = 0.
        self._prev = []

class GradModule:

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def get_params(self):
        raise NotImplementedError()