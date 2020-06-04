import time

from algorithms import rek_alg
from generators import rk_gen

m = 250
n = 100
k = 75

if __name__ == '__main__':
    start_time = time.time_ns()

    U = rk_gen.get_mat_from_normal_dist(m, k)
    V = rk_gen.get_mat_from_normal_dist(k, n)
    beta = rk_gen.get_mat_from_normal_dist(n, 1)

    X = U.dot(V)
    y = X.dot(beta)

    b = rek_alg.rk_rek(U, V, y, iterations=1_000_000, tolerance=pow(10, -4), keepErrors=True)
    rek_alg.plot_error()

    print(rek_alg.errors[0])
    print(rek_alg.errors[len(rek_alg.errors) - 1])

    end_time = time.time_ns()

    print(f"Ran in: {((end_time - start_time) / 1_000_000_000):f} seconds")
