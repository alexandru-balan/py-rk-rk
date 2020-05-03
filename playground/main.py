import time

from algorithms import rk_alg
from generators import rk_gen

m = 200
n = 150
k = 100

if __name__ == '__main__':
    start_time = time.time_ns()

    U = rk_gen.get_mat_from_normal_dist(m, k)
    V = rk_gen.get_mat_from_normal_dist(k, n)
    beta = rk_gen.get_mat_from_normal_dist(n, 1)

    X = U.dot(V)
    y = X.dot(beta)

    b = rk_alg.rk_rk(U, V, y, iterations=100_000, tolerance=pow(10, -10), keepErrors=True)
    rk_alg.plot_error()

    print(rk_alg.errors[0])
    print(rk_alg.errors[len(rk_alg.errors) - 1])

    end_time = time.time_ns()

    print(f"Ran in: {((end_time - start_time) / 1_000_000_000):f} seconds")
