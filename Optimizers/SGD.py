from typing import Callable
import math

class SGD:
    def __init__(self, params, lr=0.1):
        """Basic SGD class that updates the weights according to the gradient descent algorithm with a static LR
        :param params: list of parameters to optimize
        :param lr: learning rate
        """
        #TODO do not unpack params, leads to bugs
        self.params = params
        self.lr = lr
        self.update = lambda v, grad: v - self.lr * grad

    def update_params(self):
        for param in self.params:
            param.update(self.update)
            param.zero_grad()

class LambdaSGD(SGD):
    def __init__(self, params, lr=0.1, lr_func: Callable[[int], float] = None):
        """Lambda SGD, which updates its lr according to a passed in function lr_func
        :param params: list of parameters to optimize
        :param lr: learning rate
        :param lr_func: optional lambda function, should take in an integer representing the step and returns a % update on the base lr
        """
        super().__init__(params, lr=lr)
        if lr_func is None:
            self.lr_func = lambda x: 1.
        else:
            self.lr_func = lr_func
        self.base_lr = self.lr
        self.t = 0

    def update_lr(self):
        self.t += 1
        self.lr = self.base_lr * self.lr_func(self.t)

    def update_params(self):
        super().update_params()
        self.update_lr()


class MomentumSGD(LambdaSGD):
    def __init__(self, params, lr=0.1, momentum=0.9, lr_func = None):
        """Momentum SGD, which keeps a list of 'momenta' for each parameter, which is a weighted accumulation of its gradients. Updates each parameter
        according to new_value = old_value + cur_velocity
        :param params: list of parameters to optimize
        :param lr: learning rate
        :param momentum: how much of last step's velocity to keep, [0,1)
        :param lr_func: optional lambda function, should take in an integer representing the step and returns a % update on the base lr
        """
        super().__init__(params, lr=lr, lr_func=lr_func)
        self.momentum = momentum
        self.velocities = [0 for _ in self.params]

    def update_param_at(self, idx):
        def updater(v, grad):
            self.velocities[idx] = self.momentum * self.velocities[idx] - self.lr * grad
            return v + self.velocities[idx]
        return updater

    def update_params(self):
        for i, param in enumerate(self.params):
            param_updater = self.update_param_at(i)
            param.update(param_updater)
            param.zero_grad()
        self.update_lr()

class NesterovSGD(MomentumSGD):
    """Nesterov SGD, which modifies Momentum SGD to use the gradient post-velocity update in accumulation. Updates each parameter with the weighted accumulated gradients.
    :param params: List of parameters to optimize
    :param lr: learning rate
    :param momentum: how much of last step's velocity to keep, [0,1)
    :param lr_func: optional lambda function, should take in an integer representing the step and returns a % update on the base lr
    """
    def __init__(self, params, lr=0.1, momentum=0.9, lr_func = None):
        super().__init__(params, lr, momentum, lr_func)
        self.param_value_cache = [0 for _ in self.params]

    def gradient_update(self):
        for i, param in enumerate(self.params):
            #cache param values, update params before gradient update
            self.param_value_cache[i] = self.params[i].get_v()
            param.update(lambda v, _: v + self.momentum * self.velocities[i])

    def update_params(self):
        #reset the parameter update made for gradient calculations
        for i, param in enumerate(self.params):
            param.update(lambda _, __: self.param_value_cache[i])
        super().update_params()

class AdaGrad(MomentumSGD):
    """AdaGrad introduces a different concept than MomentumSGD: instead of updating according to an accumulation of gradients, what if we reduce lr according to it? This way, values with a lot of accumulated gradient updates will update little, and values stuck in a valley will update a lot.
    :param params: List of parameters to optimize.
    :param lr: Learning rate
    :param lr_func: optional lambda function that should take in an integer representing the step and returns a % update on the base lr
    :param eps: numerical stabilizer to avoid division by zero
    """
    def __init__(self, params, lr=0.1, lr_func = None, eps=1e-6):
        super().__init__(params, lr=lr, lr_func=lr_func)
        self.eps = eps

    def update_velocity(self, velocity, grad):
        """Defines how to update the velocity based on the gradient"""
        return velocity + grad ** 2

    def update_param_at(self, idx):
        """AdaGrad's velocity update is different: """
        def updater(v, grad):
            self.velocities[idx] = self.update_velocity(self.velocities[idx], grad)
            weighted_lr = self.lr / (self.eps + math.sqrt(self.velocities[idx]))
            return v - weighted_lr * grad
        return updater

class RMSProp(AdaGrad):
    """RMSProp is like AdaGrad, but instead of simply adding the squared gradient at every step it does a discounted update b * v_old + (1-b)*grad**2, making it less punishing for long treks through param space.
    :param params: List of parameters to optimize.
    :param lr: Learning rate
    :param lr_func: optional lambda function that should take in an integer representing the step and returns a % update on the base lr
    :param eps: numerical stabilizer to avoid division by zero
    :param beta: discounting factor in weight update
    """
    def __init__(self, params, lr=0.1, lr_func=None, eps=1e-6, beta=0.9):
        super().__init__(params, lr=lr, lr_func=lr_func, eps=eps)
        self.beta = beta

    def update_velocity(self, velocity, grad):
        return self.beta * velocity + (1 - self.beta) * grad**2

class Adam(RMSProp):
    """The goal of this section. Adam combines the advantages of RMSProp and MomentumSGD, where updates are slowed according to the accumulated velocity but also increased according to accumulated "acceleration".
    :param params: List of parameters to optimize.
    :param lr: Learning rate
    :param lr_func: optional lambda function that should take in an integer representing the step and returns a % update on the base lr
    :param eps: numerical stabilizer to avoid division by zero
    :param beta: discounting factor in weight update
    """
    def __init__(self, params, lr=0.1, lr_func=None, eps=1e-6, beta=0.9, beta2=0.9):
        super().__init__(params, lr=lr, lr_func=lr_func, eps=eps, beta=beta)
        self.beta2 = beta2
        self.accelerations = [0 for _ in self.params]

    def update_acceleration(self, acceleration, grad):
        return self.beta2 * acceleration + (1 - self.beta2) * grad

    def update_param_at(self, idx):
        def updater(v, grad):
            self.velocities[idx] = self.update_velocity(self.velocities[idx], grad)
            self.accelerations[idx] = self.update_acceleration(self.accelerations[idx], grad)
            v_hat = self.velocities[idx] / (1 - self.beta ** (self.t + 1))
            a_hat = self.accelerations[idx] / (1 - self.beta2 ** (self.t + 1))
            weighted_lr = self.lr / (self.eps + math.sqrt(v_hat))
            return v - weighted_lr * a_hat
        return updater