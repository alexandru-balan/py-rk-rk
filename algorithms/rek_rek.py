import numpy as npy
from matplotlib import pyplot

from algorithms.__internal import __ProbabilityThread, __choose_row

errors = []


def plot_error():
    x = range(len(errors))
    y = npy.array(errors) * (10.0 ** 300)

    pyplot.scatter(x, y, marker=".", alpha=0.5, c="#CE3193")
    pyplot.ylabel(r'Value [x 10^{-300}]')
    pyplot.show()


def rek_rek(
        U: npy.ndarray,
        V: npy.ndarray,
        y: npy.ndarray,
        iterations: int = 70_000,
        beta: npy.ndarray = None
) -> npy.ndarray:
    """

    :param U: A matrix with m rows and k columns
    :param V: A matrix with k rows and n columns
    :param y: A vector with m rows and 1 column
    :param iterations: The number of iterations you want the algorithm to run for
    :param beta:
        A matrix with n rows and 1 column.
        If this parameter is given then the algorithm will compute the error against it and
        retain it in the errors variable. You can later retrieve that variable for analysis.
    :return: A matrix, b, with n rows and 1 column that approximately solves the system X*b = y, where
        X = U*V
    """

    # STEP 0.
    # Initialization of variables
    rows_U = len(U)
    cols_U = len(U[0])

    rows_V = len(V)
    cols_V = len(V[0])

    x = npy.zeros((cols_U, 1))
    z = y
    b = npy.zeros((cols_V, 1))

    # STEP 1.
    # Computing the frobenius norms of U and V
    frobenius_U = pow(npy.linalg.norm(U, 'fro'), 2)
    frobenius_V = pow(npy.linalg.norm(V, 'fro'), 2)

    # STEP 2.
    # Computing the probabilities of each row of U and V and U transposed(prob. of each column of U)
    Utr = U.T

    probs_U = []
    probs_Utr = []
    probs_V = []

    threads_U = []
    threads_Utr = []
    threads_V = []

    for i in range(rows_U):  # Starting the threads for computing the probabilities of U's rows
        thread = __ProbabilityThread(frobenius_U, U[i])
        thread.start()
        threads_U.append(thread)

    for i in range(cols_U):  # Starting the threads for computing the probabilities of U's columns
        thread = __ProbabilityThread(frobenius_U, Utr[i])
        thread.start()
        threads_Utr.append(thread)

    for i in range(rows_V):  # Starting the threads for computing the probabilities of V's rows
        thread = __ProbabilityThread(frobenius_V, V[i])
        thread.start()
        threads_V.append(thread)

    # Joining the threads
    for thread in threads_U:
        thread.join()

    for thread in threads_Utr:
        thread.join()

    for thread in threads_V:
        thread.join()

    # Retrieving the results from the threads
    for thread in threads_U:
        probs_U.append(thread.probability)

    for thread in threads_Utr:
        probs_Utr.append(thread.probability)

    for thread in threads_V:
        probs_V.append(thread.probability)

    # STEP 3.
    # Repeating the same process until we go insane
    for i in range(iterations):
        chosen_U = __choose_row(probs_U)
        chosen_Utr = __choose_row(probs_Utr)
        chosen_V = __choose_row(probs_V)

        euclidean_U = pow(npy.linalg.norm(U[chosen_U], 2), 2)
        euclidean_Utr = pow(npy.linalg.norm(Utr[chosen_Utr], 2), 2)
        euclidean_V = pow(npy.linalg.norm(V[chosen_V], 2), 2)

        z = z - ((npy.array(Utr[chosen_Utr]).dot(z)) / euclidean_Utr) * npy.array(Utr[chosen_Utr]).T
        x = x + ((y[chosen_U][0] - z[chosen_U] - npy.array(U[chosen_U]).dot(x)[0]) / euclidean_U) * npy.array(
            U[chosen_U]).T
        b = b + ((x[chosen_V][0] - npy.array(V[chosen_V]).dot(b)[0]) / euclidean_V) * npy.array(V[chosen_V]).T

        # if beta parameter is given then save the error at each iteration
        if beta is not None:
            errors.append(pow(npy.linalg.norm(b - beta, 2), 2))

    return b