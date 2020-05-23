import math

import numpy as np


class HUCalculator:
    def __init__(self, n, a, b, mu, beta, sigma, f):
        self.n = n
        self.a = a
        self.b = b
        self.mu = mu
        self.beta = beta
        self.sigma = sigma
        self.f = f

        self.x = [a + (b - a) / n * i for i in range(n + 1)]

        self.y = None
        self.y_der = None

        self.A = None
        self.l = None

        self.eis = None
        self.errors = None
        self.uN = None
        self.eN = None

        self.eNs = []
        self.uNs = []
        self.Ns = []

        self.xs = []
        self.ys = []

    def calculate(self):
        self.A = self.build_left()
        self.l = self.build_right()

        self.y = np.linalg.solve(self.A, self.l)
        self.y_der = self.calculate_derivatives()

        self.uN = float(self.y.transpose().dot(self.A).dot(self.y))

        self.eis = self.calculate_eis()
        self.eN = sum(self.eis)

        self.Ns.append(self.n)
        self.eNs.append(self.eN)
        self.uNs.append(self.uN)

        self.xs.append(self.x)
        self.ys.append(self.y)

        return self

    def calculate_derivatives(self):
        y_der = np.zeros(len(self.y))

        for i in range(1, len(self.y)):
            y_der[i - 1] = (self.y[i] - self.y[i - 1]) / (self.x[i] - self.x[i - 1])

        return y_der

    def calc_until(self, accuracy, max_steps=10):
        prev = self.next_step(accuracy).n
        i = 0
        while i < max_steps:
            i += 1

            self.next_step(accuracy)

            if prev != self.n:
                prev = self.n
            else:
                break

        return self

    def next_step(self, accuracy):
        if self.y is None:
            return self.calculate()

        extended_x = []

        self.errors = np.zeros(self.n)
        for i in range(self.n):
            self.errors[i] = self.calculate_error_at_i(i)
            if self.errors[i] > accuracy:
                extended_x.append(self.x[i])
                extended_x.append(self.x[i] + (self.x[i + 1] - self.x[i]) / 2)
            else:
                extended_x.append(self.x[i])

        extended_x.append(self.x[self.n])

        if len(extended_x) == self.n + 1:
            return self

        self.x = extended_x
        self.n = len(extended_x) - 1

        return self.calculate()

    def calculate_error_at_i(self, i):
        ei_sq = math.sqrt(self.eis[i])

        return (ei_sq * math.sqrt(self.n) * 100) / math.sqrt(self.uN + self.eN)

    def calculate_eis(self):
        if self.eis is not None and len(self.eis) == self.n:
            return self.eis

        e = np.ones(self.n)
        for i in range(self.n):
            e[i] = self.calculate_ei(i)

        return e

    def calculate_ei(self, i):
        h = self.x[i + 1] - self.x[i]
        xi = self.x[i]
        xip = xi + h / 2

        m = h ** 3 / self.mu(xip)
        b = self.f(xip) - self.beta(xip) * self.y_der[i] - self.sigma(xip) * self.y[i]
        d = 10 + (h * self.beta(xip) / self.mu(xip)) * (h * h * self.sigma(xip) / self.mu(xip))

        return math.fabs(5 / 6 * m * b * b / d)

    def build_left(self):
        A = np.identity(self.n + 1)
        for i in range(0, self.n):
            hp = self.x[i + 1] - self.x[i]
            hm = self.x[i] - self.x[i - 1]

            xi = self.x[i]
            xip = xi + hp / 2
            xim = xi - hm / 2

            A[i][i] = 1 / hm * self.mu(xim) + 1 / hp * self.mu(xip) + \
                      0.5 * self.beta(xim) - 0.5 * self.beta(xip) + \
                      hm / 3 * self.sigma(xim) + hp / 3 * self.sigma(xip)

            A[i][i + 1] = -1 / hp * self.mu(xip) + 0.5 * self.beta(xip) + hp / 6 * self.sigma(xip)
            A[i + 1][i] = -1 / hp * self.mu(xip) - 0.5 * self.beta(xip) + hp / 6 * self.sigma(xip)

        A[self.n][self.n] = pow(10, 10)
        A[0][0] = pow(10, 10)

        return A

    def build_right(self):
        l = np.zeros((self.n + 1, 1))
        for i in range(1, self.n):
            hp = self.x[i + 1] - self.x[i]
            hm = self.x[i] - self.x[i - 1]

            xi = self.x[i]
            xip = xi + hp / 2
            xim = xi - hm / 2

            l[i] = hm / 2 * self.f(xim) + hp / 2 * self.f(xip)

        h0p = (self.x[1] - self.x[0])
        l[0] = 0.5 * h0p * self.f(self.a + h0p)

        hnm = (self.x[self.n] - self.x[self.n - 1])
        l[self.n] = 0.5 * hnm * self.f(self.b - hnm)

        return l

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def get_errors(self):
        return self.errors
