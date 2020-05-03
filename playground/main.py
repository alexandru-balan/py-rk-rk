import time

import numpy as npy

from algorithms import rk_alg, rek_alg
from generators import rk_gen

m = 5
n = 3
k = 2

if __name__ == '__main__':
    start_time = time.time_ns()

    U = rk_gen.get_mat_from_normal_dist(m, k)
    V = rk_gen.get_mat_from_normal_dist(k, n)
    beta = rk_gen.get_mat_from_normal_dist(n, 1)

    X = U.dot(V)
    y = X.dot(beta)

    beta = npy.linalg.lstsq(X, y, rcond=None)[0]

    rek_alg.rek_rek(U, V, y, beta=beta, iterations=10000)
    rek_alg.plot_error()

    print(rek_alg.errors[0])
    print(rek_alg.errors[len(rk_alg.errors) - 1])

    end_time = time.time_ns()

    print(f"Ran in: {((end_time - start_time) / 1_000_000_000):f} seconds")
