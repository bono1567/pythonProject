import numpy as np
import matplotlib.pyplot as plt


def heston_model_simulation(S0, v0, rho, kappa, theta, sigma, T, N, M):
    dt = T / N
    mu = np.array([0, 0])
    cov = np.array([[1, rho],
                    [rho, 1]])

    S = np.full(shape=(N + 1, M), fill_value=S0)
    v = np.full(shape=(N + 1, M), fill_value=v0)
    m = np.full(shape=(N + 1, M), fill_value=0.2)

    Z = np.random.multivariate_normal(mu, cov, (N, M))
    for i in range(1, N + 1):
        S[i] = S[i - 1] * np.exp((r - q - 0.5 * v[i - 1]) * dt + np.sqrt(v[i - 1] * dt) * Z[i - 1, :, 0])
        v[i] = np.maximum(v[i - 1] + kappa * (theta - v[i - 1]) * dt + sigma * np.sqrt(v[i - 1] * dt) * Z[i - 1, :, 1],
                          0)
        m[i] = m[i - 1]*np.exp(-theta * Z[i - 1, :, 0])

    return S, v, m


if __name__ == "__main__":
    T = 1
    r = 0.01
    q = 0.02
    S0 = 2300
    kappa = 0.7
    rho_a = -0.60
    M = 5
    N = 252
    v0 = 0.20 ** 2
    theta = 0.15 ** 2
    sigma = 0.5

    S_n, v_n, M_n = heston_model_simulation(S0, v0, rho_a, kappa, theta, sigma, T, N, M)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 5))
    time = np.linspace(0, T, N + 1)
    ax1.plot(time, S_n)
    ax1.set_title('Asset Prices')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Asset Prices')
    ax2.plot(time, v_n)
    ax2.set_title('Volatility')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Volatility')
    ax3.plot(time, M_n)
    ax3.set_title('Risk Price')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Risk Price')
    plt.show()
