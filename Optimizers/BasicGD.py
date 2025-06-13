import random
from GradGraph import Variable, GradModule
from typing import Callable


class MonomialParam:
    def __init__(self, coeff, degree):
        self.coeff = Variable(coeff)
        self.degree = degree

    def get_v(self):
        return self.coeff.v

    def forward(self, x: Variable):
        return self.coeff * (x ** self.degree)

    def __call__(self, x):
        x = Variable(x) if not isinstance(x, Variable) else x
        return self.forward(x)

    def update(self, lr: Callable[[float, float], float]):
        self.coeff.update_v(lr)

    def zero_grad(self):
        self.coeff.zero_grad()

class Polynomial:
    def __init__(self, coefficients: list[float] = None, degrees: list[float] = None, num_terms: int = None):
        if coefficients is None and degrees is None and num_terms is None:
            raise ValueError("Either coefficients or degrees or num_terms must be specified")
        if coefficients is not None and num_terms is None:
            num_terms = len(coefficients)
        if degrees is not None and num_terms is None:
            num_terms = len(degrees)

        if coefficients is None:
            coefficients = [random.random() for _ in range(num_terms)]
        if degrees is None:
            degrees = list(range(1, num_terms + 1)) #currently not dealing with bias

        self.monomials = [MonomialParam(coeff=coeff, degree=degree) for coeff, degree in zip(coefficients, degrees)]

    def __call__(self, x):
        result = Variable(0)
        for monomial in self.monomials:
            result += monomial(x)
        return result

    def get_params(self):
        return self.monomials


