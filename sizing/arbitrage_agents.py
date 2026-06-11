"""Dispatch agents for PriceEnv (hourly arbitrage).

Three policies, all perfect-foresight or forecast-driven, all open-loop
(solve once over the horizon, apply actions).

  lp_linear_actions   : linear cycling cost (mu * |a|).  scipy.optimize.linprog.
  qp_quadratic_actions: quadratic cycling cost (alpha * a^2).  SLSQP via
                        scipy.optimize.minimize.
  qp_ensemble_actions : K-forecast averaging into a single QP solve.
                        Cheap stochastic-LP proxy.

Sign convention: action a > 0 = discharge, a < 0 = charge (matches PriceEnv).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import (Bounds, LinearConstraint, linprog, minimize)


def _soc_constraint_matrices(T: int, b_E: float, soc0: float, eta: float):
    """Build A x <= b for SoC bounds, where x = [P_chg, P_dis] >= 0.

    SoC[t] = soc0 + eta * sum(P_chg[0..t]) - sum(P_dis[0..t]) / eta,
    constrained to [0, b_E].

    Vectorized sparse construction via scipy.sparse.tril for O(T^2/2) nnz.
    """
    from scipy.sparse import bmat, csr_matrix, tril
    inv_eta = 1.0 / max(eta, 1e-9)
    L = tril(np.ones((T, T)), format="csr")  # lower-triangular ones
    # Upper SoC bound: eta*L * P_chg - inv_eta*L * P_dis <= b_E - soc0
    A_upper = bmat([[eta * L, -inv_eta * L]], format="csr")
    # Lower SoC bound: -eta*L * P_chg + inv_eta*L * P_dis <= soc0
    A_lower = bmat([[-eta * L, inv_eta * L]], format="csr")
    A = bmat([[A_upper], [A_lower]], format="csr")
    b = np.concatenate([
        np.full(T, b_E - soc0),
        np.full(T, soc0),
    ])
    return A, b


def lp_linear_actions(prices, b_E: float, b_P: float, soc0: float,
                      mu: float = 0.0, eta: float = 1.0) -> np.ndarray:
    """Solve LP-linear, return action vector length T (signed: + = discharge).

    Variables x = [P_chg[0..T-1], P_dis[0..T-1]] >= 0.
    Objective: minimize -prices.dot(P_dis - P_chg) + mu * sum(P_chg + P_dis)
    """
    prices = np.asarray(prices, dtype=float)
    T = len(prices)
    # State-variable formulation: x = [P_chg, P_dis, soc]. Equivalent to the
    # cumulative-sum form but with O(T) constraint nonzeros instead of
    # O(T^2), which keeps HiGHS factorization memory flat (~MBs, not ~GB).
    from scipy.sparse import bmat, eye, identity
    inv_eta = 1.0 / max(eta, 1e-9)
    c = np.concatenate([prices + mu, -prices + mu, np.zeros(T)])
    bounds = [(0.0, b_P)] * (2 * T) + [(0.0, b_E)] * T
    # soc[t] - soc[t-1] - eta*P_chg[t] + inv_eta*P_dis[t] = (soc0 if t==0 else 0)
    D = identity(T, format="csr") - eye(T, k=-1, format="csr")
    A_eq = bmat([[-eta * identity(T), inv_eta * identity(T), D]], format="csr")
    b_eq = np.zeros(T)
    b_eq[0] = soc0
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"lp_linear_actions: {res.message}")
    P_chg, P_dis = res.x[:T], res.x[T:2 * T]
    return P_dis - P_chg


def lp_maxmin_actions(members, b_E: float, b_P: float, soc0: float,
                      mu: float = 0.0, eta: float = 1.0) -> np.ndarray:
    """Worst-case-robust LP: maximize min_k (members[k] @ a - mu * cycling).

    A risk-averse alternative to planning on the ensemble mean (which,
    for a linear objective, is the expectation-optimal stochastic plan).
    Variables x = [P_chg, P_dis, soc, t]; maximize t subject to
    t <= revenue_k for every ensemble member k, plus battery dynamics
    (same state-variable formulation as lp_linear_actions).
    """
    from scipy.sparse import bmat, csr_matrix, eye, identity
    members = np.asarray(members, dtype=float)
    K, T = members.shape
    inv_eta = 1.0 / max(eta, 1e-9)
    c = np.zeros(3 * T + 1)
    c[-1] = -1.0  # maximize t
    bounds = ([(0.0, b_P)] * (2 * T) + [(0.0, b_E)] * T + [(None, None)])
    D = identity(T, format="csr") - eye(T, k=-1, format="csr")
    A_eq = bmat([[-eta * identity(T), inv_eta * identity(T), D,
                  csr_matrix((T, 1))]], format="csr")
    b_eq = np.zeros(T)
    b_eq[0] = soc0
    # t - (p_k . (P_dis - P_chg) - mu . (P_chg + P_dis)) <= 0
    A_ub = np.zeros((K, 3 * T + 1))
    for k in range(K):
        A_ub[k, :T] = members[k] + mu
        A_ub[k, T:2 * T] = -members[k] + mu
        A_ub[k, -1] = 1.0
    res = linprog(c, A_ub=csr_matrix(A_ub), b_ub=np.zeros(K),
                  A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"lp_maxmin_actions: {res.message}")
    return res.x[T:2 * T] - res.x[:T]


def qp_quadratic_actions(prices, b_E: float, b_P: float, soc0: float,
                         alpha: float = 0.0, eta: float = 1.0) -> np.ndarray:
    """Solve QP with quadratic cycling cost via cvxpy + OSQP/CLARABEL.

    Objective: maximize prices.dot(P_dis - P_chg) - alpha * sum (P_chg + P_dis)^2

    cvxpy lets us express the QP in natural form; OSQP / CLARABEL solves it
    in <1 s for T~200. ~50x faster than scipy.optimize SLSQP / trust-constr.
    """
    import cvxpy as cp
    prices = np.asarray(prices, dtype=float)
    T = len(prices)
    P_chg = cp.Variable(T, nonneg=True)
    P_dis = cp.Variable(T, nonneg=True)
    cycling = cp.sum_squares(P_chg + P_dis)
    rev = prices @ (P_dis - P_chg)
    obj = cp.Maximize(rev - alpha * cycling)
    cons = [P_chg <= b_P, P_dis <= b_P]
    inv_eta = 1.0 / max(eta, 1e-9)
    soc_t = soc0 + eta * cp.cumsum(P_chg) - inv_eta * cp.cumsum(P_dis)
    cons += [soc_t >= 0, soc_t <= b_E]
    prob = cp.Problem(obj, cons)
    try:
        prob.solve(solver="CLARABEL")
    except Exception:
        prob.solve(solver="SCS")
    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"qp_quadratic_actions: status {prob.status}")
    return np.asarray(P_dis.value - P_chg.value)


def qp_ensemble_actions(forecasts, b_E: float, b_P: float, soc0: float,
                        alpha: float = 0.0, eta: float = 1.0) -> np.ndarray:
    """Average K forecasts into a mean, solve QP-quadratic on the mean.

    forecasts : array of shape (K, T).  Returns action vector length T.

    Cheap stochastic-LP proxy. By Jensens inequality + convex cost, generally
    beats single-forecast QP under noise.
    """
    forecasts = np.asarray(forecasts, dtype=float)
    if forecasts.ndim != 2:
        raise ValueError("forecasts must be (K, T)")
    forecast_mean = forecasts.mean(axis=0)
    return qp_quadratic_actions(forecast_mean, b_E, b_P, soc0, alpha, eta)


# -----------------------------------------------------------------------------
# Convenience: run an action vector on a PriceEnv, return summary.
# -----------------------------------------------------------------------------
def run_actions(env, actions) -> dict:
    """Apply a precomputed action vector to env, return realized (R, D, soc_log)."""
    actions = np.asarray(actions, dtype=float)
    if len(actions) < env.T:
        raise ValueError(f"actions length {len(actions)} < env.T {env.T}")
    env.reset()
    for t in range(env.T):
        env.step(actions[t])
    R = float(np.sum(env.prices[:env.T] * np.asarray(env.action_log)))
    return {
        "R": R,
        "soc_log": np.asarray(env.soc_log),
        "actions": np.asarray(env.action_log),
        "rewards": np.asarray(env.reward_log),
    }
