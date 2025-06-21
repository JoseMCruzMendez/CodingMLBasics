import numpy as np
from typing import Callable, Literal, Dict
from NpAutograd import Tensor, GradModule

class Regularizer:
    """Base class for regularizers"""
    def __init__(self, penalty: float = 1e-3):
        """
        :param penalty: penalty coefficient for regularization term, defaults to 1e-3
        """
        self.penalty = penalty

    def _acc(self, w: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def grad(self, w: np.ndarray)->float:
        out = self._acc(w)
        return out * self.penalty

    def __call__(self, w: np.ndarray)->float:
        return self.grad(w)

class L1(Regularizer):
    def _acc(self, w: np.ndarray) -> np.ndarray:
        return np.sign(w)

class L2(Regularizer):
    def _acc(self, w: np.ndarray):
        return w

class Linf(Regularizer):
    def _acc(self, w: np.ndarray):
        g = np.zeros_like(w)
        idx = np.unravel_index(np.abs(w).argmax(), w.shape)
        g[idx] = self.penalty * np.sign(w[idx])
        return g

class LinfFull(Regularizer):
    def _acc(self, w: np.ndarray):
        idx = np.unravel_index(np.abs(w).argmax(), w.shape)
        g = self.penalty * np.sign(w[idx])
        return np.full_like(w, g)

class GlobalLinf:
    """Global regularizer, penalizes on the max of the absolute value of all weights."""
    def __init__(self, penalty: float = 1e-3):
        self.penalty = penalty

#TODO finish global inf
    def grad(self, w: list[GradModule])->float:
        max = -np.inf
        sgn = 0
        for module in w:
            for param in module.get_params():
                argmax_idx = np.argmax(np.abs(param.v))
        return self.penalty


_REG_TYPES: Dict[str, Regularizer] = {
    "l1"  : L1,
    "l2"  : L2,
    "linf": Linf,
    "glinf": GlobalLinf,
    "flinf": LinfFull
}


class SGD:
    """Now with reg and lambda built-in!"""
    def __init__(self, params:list[Tensor],
                 lr:float=0.1,
                 lr_func:Callable[[int], float] = None,
                 reg_type: Literal["L1", "L2", "Linf", "FLinf"]="L1",
                 reg_param: float= 1e-3):
        self.params = params
        self.lr = lr
        self.lr_func = lr_func
        self.reg: Regularizer = _REG_TYPES[reg_type.lower()](penalty=reg_param)
        if lr_func is None:
            self.lr_func = lambda _: 1.
        else:
            self.lr_func = lr_func

        self.t: int = 0


    #Trying a less closure-heavy approach
    def step(self):
        lr_t = self.lr * self.lr_func(self.t)
        for param in self.params:
            reg = self.reg(param.v)
            param.update_v(lambda w, g, lr=lr_t, rg=reg: self._sgd_rule(w, g, lr, rg))
            param.zero_grad()
        self.t += 1

    # -------- internals -----------------------------------------------------
    def _sgd_rule(self, w: np.ndarray, grad: np.ndarray, lr: float, reg: float):
        grad = np.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)          # weight-decay term
        return w - lr * (grad + reg)

class MomentumSGD(SGD):
    """Momentum SGD. With all accumulation methods I will implement a W equivalent that does not accumulate regularization penalties and instead adds them at update time only"""
    def __init__(self,
                 *SGDParams,
                 momentum:float=0.9,
                 **SGDKwargs):
        super().__init__(*SGDParams, **SGDKwargs)
        self.momentum = momentum
        #This is probably not good coding practice but I will run with this
        self.velocities: Dict[int, np.ndarray] = {}
        self.i = 0
        for param in self.params:
            self.velocities[self.i] = np.zeros_like(param.v)
            self.i += 1
        self.i = 0

    def _sgd_rule(self, v, grad, lr, reg):
        self.velocities[self.i] = self.momentum * self.velocities[self.i] - lr * (grad + reg)
        return v + self.velocities[self.i]

    def step(self):
        lr_t = self.lr * self.lr_func(self.t)
        for i, param in enumerate(self.params):
            self.i = i
            reg = self.reg(param.v)
            param.update_v(lambda w, g, lr=lr_t, rg=reg: self._sgd_rule(w, g, lr, rg))
            param.zero_grad()
        self.t += 1

class MomentumSGDW(MomentumSGD):
    def _sgd_rule(self, v, grad, lr, reg):
        self.velocities[self.i] = self.momentum * self.velocities[self.i] - lr * grad
        return v + self.velocities[self.i] - lr * reg

class AdaGrad(MomentumSGD):
    def __init__(self,
                 *args,
                 eps=1e-12,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.eps = eps

    def _sgd_rule(self, v, grad, lr, reg):
        self.velocities[self.i] += (grad + reg) ** 2
        return v - lr * grad / (np.sqrt(self.velocities[self.i]) + self.eps)

class AdaGradW(AdaGrad):
    def _sgd_rule(self, v, grad, lr, reg):
        self.velocities[self.i] += (grad) ** 2
        #notice how reg is not accumulated, instead just added inside lr
        return v - self.lr * ((grad / (np.sqrt(self.velocities[self.i]) + self.eps)) + reg)

class RMSProp(AdaGrad):
    def _sgd_rule(self, v, grad, lr, reg):
        #EMS of gradients squared
        self.velocities[self.i] = self.momentum * self.velocities[self.i] + (1-self.momentum) * (grad + reg)**2
        return v - self.lr * grad / (np.sqrt(self.velocities[self.i]) + self.eps)

class RMSPropW(RMSProp):
    def _sgd_rule(self, v, grad, lr, reg):
        self.velocities[self.i] = self.momentum * self.velocities[self.i] + (1-self.momentum) * grad ** 2
        return v - self.lr * ((grad / (np.sqrt(self.velocities[self.i]) + self.eps)) + reg)

class Adam(RMSProp):
    def __init__(self, *args, beta2 = 0.99, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta2 = beta2
        self.accelerations: Dict[int, np.ndarray] = {}
        for param in self.params:
            self.accelerations[self.i] = np.zeros_like(param.v)
            self.i += 1
        self.i = 0
    def _sgd_rule(self, v, grad, lr, reg):
        self.velocities[self.i] = self.momentum * self.velocities[self.i] + (1-self.momentum) * (grad + reg)**2
        self.accelerations[self.i] = self.beta2 * self.accelerations[self.i] + (1-self.beta2) * (grad + reg)
        v_hat = self.velocities[self.i]  / (1 - self.momentum ** (self.t + 1))
        a_hat = self.accelerations[self.i] / (1 - self.beta2 ** (self.t + 1))
        return v - lr * (a_hat / (np.sqrt(v_hat) + self.eps))

class AdamW(Adam):
    #Made it one step up
    def _sgd_rule(self, v, grad, lr, reg):
        self.velocities[self.i] = self.momentum * self.velocities[self.i] + (1-self.momentum) * grad ** 2
        self.accelerations[self.i] = self.beta2 * self.accelerations[self.i] + (1-self.beta2) * grad
        v_hat = self.velocities[self.i] / (1 - self.momentum ** (self.t + 1))
        a_hat = self.accelerations[self.i] / (1 - self.beta2 ** (self.t+1))
        return v - lr * ((a_hat/(np.sqrt(v_hat) + self.eps)) + reg)