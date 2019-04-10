import time
import numpy as np
import scipy.linalg as la


Kopt = None


def process(A, B, x, u):
    x = A * x + B *u
    return x


def solve_DARE_with_iteration(A, B, Q, R):
    """
    solve a discrete time_Algebraic Riccati equation (DARE)
    """
    X = Q
    maxiter = 150
    eps = 0.01

    for i in range(maxiter):
        Xn = A.T * X * A - A.T * X * B * \
            la.inv(R + B.T * X * B) * B.T * X * A + Q
        if (abs(Xn - X)).max() < eps:
            X = Xn
            break
        X = Xn

    return Xn


def dlqr_with_iteration(Ad, Bd, Q, R):
    """Solve the discrete time lqr controller.
    x[k+1] = Ad x[k] + Bd u[k]
    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    # ref Bertsekas, p.151
    """

    # first, try to solve the ricatti equation
    X = solve_DARE_with_iteration(Ad, Bd, Q, R)

    # compute the LQR gain
    K = np.matrix(la.inv(Bd.T * X * Bd + R) * (Bd.T * X * Ad))

    return K


def dlqr_with_arimoto_potter(Ad, Bd, Q, R, dt):
    """Solve the discrete time lqr controller.
    x[k+1] = Ad x[k] + Bd u[k]
    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    # ref Bertsekas, p.151
    """

    n = len(Bd)

    # continuous
    Ac = (Ad - np.eye(n)) / dt
    Bc = Bd / dt

    # Hamiltonian
    Ham = np.vstack(
        (np.hstack((Ac, - Bc * la.inv(R) * Bc.T)),
         np.hstack((-Q, -Ac.T))))

    eigVals, eigVecs = la.eig(Ham)

    V1 = None
    V2 = None

    for i in range(2 * n):
        if eigVals[i].real < 0:
            if V1 is None:
                V1 = eigVecs[0:n, i]
                V2 = eigVecs[n:2 * n, i]
            else:
                V1 = np.vstack((V1, eigVecs[0:n, i]))
                V2 = np.vstack((V2, eigVecs[n:2 * n, i]))
    V1 = np.matrix(V1.T)
    V2 = np.matrix(V2.T)

    P = (V2 * la.inv(V1)).real

    K = la.inv(R) * Bc.T * P

    return K


def lqr_ref_tracking(x, xref, uref, A, B, Q, R, dt):
    global Kopt
    if Kopt is None:
        #  start = time.time()
        #  Kopt = dlqr_with_iteration(A, B, np.eye(2), np.eye(1))
        Kopt = dlqr_with_arimoto_potter(A, B, Q, R, dt)

        #  elapsed_time = time.time() - start
        #  print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

    u = -uref - Kopt * (x - xref)

    return u

