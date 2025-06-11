import numpy as np
import cvxpy as cp
import time
from typing import Tuple

def solve_quadprog(P, q, Gs, hs, z):
    prob = cp.Problem(
        cp.Minimize((1/2) * cp.quad_form(z, P) + q.T @ z),
        [Gi @ z <= hi for (Gi, hi) in zip(Gs, hs)]
    )
    prob.solve()
    if prob.status != "optimal":
        print(prob.status)
        raise RuntimeError("no solution found to quadratic program. aborting.")
    return z.value

def solve_controller(grad_h, h, f, g, u_n, x, u_bounds: Tuple):
    x = np.squeeze(x)
    is_vectorized = False
    if len(x.shape) > 1:
        is_vectorized = True

    results = None
    if is_vectorized:

        # wrapper to call solve_controller with fixed args
        def controller_wrapper(x):
            return _solve_controller(grad_h, h, f, g, u_n, x, u_bounds)

        # convert xs to shape (k, 2) rather than (2, k)
        x = x.T  # shape becomes (k, 2), suitable for iteration
        return np.array([controller_wrapper(xi) for xi in x])
    else:
        return _solve_controller(grad_h, h, f, g, u_n, x, u_bounds)

    return results  # list of np.ndarray, each is a control u

def _solve_controller(grad_h, h, f, g, u_n, x, u_bounds: Tuple):
    x = np.atleast_1d(x)
    u_lb, u_ub = u_bounds
    alpha = 1e4
    epsilon = 1e-4

    M = 1000.0

    P = np.array([
        [1, 0],
        [0, 0],
    ])
    q = np.array([
        [-u_n(x).item()],
        [M],
    ])
    # (partial h) / (partial x) f(x) + (partial h) / (partial x) g(x) u >= alpha(h(x))
    cat = np.concatenate
    Gs = [
        cat((-1 * grad_h(x).T @ g(x)[:, None], -np.ones((4,1))), axis=1), # slack variable here
        np.array([0, -1]),
        np.array([-1, 0]),
        np.array([1, 0]),
    ]
    hs = [
        alpha * h(x) + grad_h(x).T @ f(x), # slack variable here
        0,
        -u_lb,
        u_ub,
    ]
    z = cp.Variable(2) # z = [u, s]
    try:
        u, s = solve_quadprog(P, q, Gs, hs, z)
    except Exception as e:
        print(f"{x=}, {u_bounds=}")
        raise e
    return np.atleast_1d(u)
