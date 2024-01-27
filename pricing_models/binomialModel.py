import math
import numpy as np
import matplotlib.pyplot as plt


class OptionModel:

    def __init__(self, T, r, variance, s_0, k, n, option_type="C", op_type="EO", show=False):
        self.cut_off = []
        self.T = T
        self.r = r
        self.op_type = op_type
        self.type = option_type
        self.strike = k
        self.period = n + 1
        self.delta_t = self.T / n
        self.volatility = variance
        self.s_0 = s_0
        self.upper = math.exp(self.volatility * math.sqrt(self.delta_t))
        self.lower = 1 / self.upper
        self.R = math.exp(self.r * self.delta_t)
        self.qu = (self.R - self.lower) / (self.upper - self.lower)
        self.qd = 1 - self.qu
        print("VECTORIZED: ",
              round(american_fast_tree(self.strike, 0.5, self.s_0, self.r, self.period, self.upper, self.lower), 5))
        # print("LOOPS: ", round(self.p_matrix[0][0], 5))
        if show:
            plt.plot([i for i, _ in enumerate(self.cut_off)], self.cut_off)
            plt.show()

    def create_stock_matrix(self):
        matrix = np.zeros((self.period, self.period))
        for i, _ in enumerate(matrix[0]):
            if i == 0:
                matrix[0][i] = self.s_0
            else:
                matrix[0][i] = matrix[0][i - 1] * self.upper

        for i in range(1, self.period):
            for j in range(i, self.period):
                matrix[i][j] = matrix[i - 1][j - 1] * self.lower

        return matrix

    def create_payoff_matrix(self):
        matrix = np.zeros((self.period, self.period))
        for i in range(self.period):
            if self.type == "C":
                matrix[i][-1] = max(self.s_prices[i][-1] - self.strike, 0)
            else:
                matrix[i][-1] = max(self.strike - self.s_prices[i][-1], 0)
        for j in range(self.period - 2, -1, -1):
            price_at = []
            for i in range(j + 1):
                matrix[i][j] = (self.qu * matrix[i][j + 1] + self.qd * matrix[i + 1][j + 1]) / \
                               math.exp(self.r * self.delta_t)
                if self.op_type == "AO":
                    if self.type == "C":
                        if self.s_prices[i][j] - self.strike <= matrix[i][j]:
                            price_at.append(self.s_prices[i][j])
                        matrix[i][j] = max(matrix[i][j], self.s_prices[i][j] - self.strike)
                    else:
                        if self.strike - self.s_prices[i][j] <= matrix[i][j]:
                            price_at.append(self.s_prices[i][j])
                        matrix[i][j] = max(matrix[i][j], self.strike - self.s_prices[i][j])
            if price_at:
                self.cut_off.append(price_at[0])
        return matrix

    def binomial_tree_vec(self):
        if self.type == "C":
            payoff = [x - self.strike for x in self.s_prices[:, -1]]
        else:
            payoff = [self.strike - x for x in self.s_prices[:, -1]]
        payoff = np.maximum(0, payoff)
        if self.op_type != "AO":
            state_n = np.linalg.matrix_power(self.state_prices, self.period - 1)
            final_payoff = np.matmul(state_n, payoff)
            return final_payoff[0]

        for i in np.arange(1, self.period):
            payoff = np.matmul(self.state_prices, payoff)
            if self.type == 'P':
                payoff = np.maximum(payoff, self.strike - self.s_prices[:, self.period - i - 1])
            else:
                payoff = np.maximum(payoff, self.s_prices[:, self.period - i - 1] - self.strike)
        return payoff[0]

    def create_state_prices(self):
        matrix = np.zeros((self.period, self.period))
        for i in range(self.period):
            if i + 1 < self.period:
                matrix[i][i + 1] = self.qd / self.R
            matrix[i][i] = self.qu / self.R
        return matrix


def american_fast_tree(K, T, S0, r, N, u, d, opttype='P'):
    # precompute values
    dt = T / N
    q = (np.exp(r * dt) - d) / (u - d)
    disc = np.exp(-r * dt)

    # initialise stock prices at maturity
    S = S0 * d ** (np.arange(N, -1, -1)) * u ** (np.arange(0, N + 1, 1))

    # option payoff
    if opttype == 'P':
        C = np.maximum(0, K - S)
    else:
        C = np.maximum(0, S - K)

    # backward recursion through the tree
    for i in np.arange(N - 1, -1, -1):
        S = S0 * d ** (np.arange(i, -1, -1)) * u ** (np.arange(0, i + 1, 1))
        C[:i + 1] = disc * (q * C[1:i + 2] + (1 - q) * C[0:i + 1])
        C = C[:-1]
        if opttype == 'P':
            C = np.maximum(C, K - S)
        else:
            C = np.maximum(C, S - K)

    return C[0]


def run():
    OptionModel(0.5, 0.04, 0.3, 40, 40, 1000, option_type="P", op_type="AO", show=True)


if __name__ == '__main__':
    # OptionModel(0.5, 0.04, 0.3, 40, 40, 50, option_type="P", op_type="AO")
    # OptionModel(0.5, 0.04, 0.3, 40, 40, 3000, option_type="P", op_type="AO")
    OptionModel(0.5, 0.04, 0.3, 40, 40, 100000, option_type="P", op_type="AO", show=False)
