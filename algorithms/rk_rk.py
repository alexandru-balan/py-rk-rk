import matplotlib.pyplot as pyplot
import numpy as npy

from algorithms.__internal import __ProbabilityThread, __choose_row

errors = []

THREAD_NUMBER = 8


def plot_error():
    x = range(len(errors))
    y = errors

    pyplot.scatter(x, y, marker=".", alpha=0.5, c="#1f77b4")
    pyplot.show()


def rk_rk(U: npy.ndarray,
          V: npy.ndarray,
          y: npy.ndarray,
          iterations: int = 70_000,
          tolerance: float = pow(10, -4),
          keepErrors: bool = False) -> npy.ndarray:
    """
    :param U: A matrix with m rows and k columns
    :param V: A matrix with k rows and n columns
    :param y: A vector with m rows and 1 column
    :param iterations: The number of iterations you want the algorithm to run for
    :param tolerance:
    :param keepErrors:
    :return: A matrix, b, with n rows and 1 column that approximately solves the system X*b = y, where
        X = U*V
    """
    rows_U = len(U)
    cols_U = len(U[0])

    rows_V = len(V)
    cols_V = len(V[0])

    x = npy.zeros((cols_U, 1))
    b = npy.zeros((cols_V, 1))

    # STEP 1.
    # Computing the squared Frobenuis norms of U and V
    frobenius_U = pow(npy.linalg.norm(U, 'fro'), 2)
    frobenius_V = pow(npy.linalg.norm(V, 'fro'), 2)

    # STEP 2.
    # Computing the probability of each row in multi-threaded manner
    probs_U = []
    probs_V = []

    threads_U = []
    threads_V = []

    for i in range(0, rows_U, THREAD_NUMBER):  # Starting the threads for computing the probabilities of U's rows
        if i <= rows_U - THREAD_NUMBER:
            thread = __ProbabilityThread(frobenius_U, U[i:i + THREAD_NUMBER])
        else:
            thread = __ProbabilityThread(frobenius_U, U[i:])
        thread.start()
        threads_U.append(thread)

    for i in range(0, rows_V, THREAD_NUMBER):  # Starting the threads for computing the probabilities of V's rows
        if i <= rows_V - THREAD_NUMBER:
            thread = __ProbabilityThread(frobenius_V, V[i:i + THREAD_NUMBER])
        else:
            thread = __ProbabilityThread(frobenius_V, V[i:])
        thread.start()
        threads_V.append(thread)

    # Joining the threads
    for thread in threads_U:
        thread.join()

    for thread in threads_V:
        thread.join()

    # Retrieving the results from the threads
    for thread in threads_U:
        for probability in thread.probability:
            probs_U.append(probability)

    for thread in threads_V:
        for probability in thread.probability:
            probs_V.append(probability)

    # STEP 3.
    # Repeating the same process until we go insane
    for i in range(iterations):
        chosen_U = __choose_row(probs_U)
        chosen_V = __choose_row(probs_V)

        euclidean_U = pow(npy.linalg.norm(U[chosen_U], 2), 2)
        euclidean_V = pow(npy.linalg.norm(V[chosen_V], 2), 2)

        x = x + ((y[chosen_U][0] - npy.array(U[chosen_U]).dot(x)[0]) / euclidean_U) * npy.array(U[chosen_U]).T
        b = b + ((x[chosen_V][0] - npy.array(V[chosen_V]).dot(b)[0]) / euclidean_V) * npy.array(V[chosen_V]).T

        # if beta parameter is given then save the error at each iteration
        if keepErrors:
            err1 = V.dot(b)
            err2 = U.dot(err1)
            errors.append(pow(npy.linalg.norm(err2 - y, 2), 2))
            if errors[i] <= tolerance:
                break

    return b
