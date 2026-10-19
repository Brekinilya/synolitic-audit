"""Batched pairwise logistic regression, numerically equivalent to sklearn.

Fits one L2-regularised logistic model per node pair at once with damped Newton
steps, chunked over pairs. Matches
``LogisticRegression(C=C, fit_intercept=True, penalty='l2')``, whose objective is

    sum_i log-loss_i + (1 / (2C)) * ||w||^2        (intercept unpenalised)

which is what the published ensemble-graph construction uses with C = 1.
"""
from __future__ import annotations

import numpy as np

KINDS = {"mmssr": ("m", "s", "r"), "mmss": ("m", "s"), "r": ("r",)}


def n_features(kind: str) -> int:
    return 2 * ("m" in kind) + 2 * ("s" in kind) + 1 * ("r" in kind)


def design(vertex, edge, pi, pj, kind):
    """(n, K, q+1) design block for the given pairs, intercept last."""
    cols = []
    if "m" in kind:
        cols += [vertex[:, pi, 0], vertex[:, pj, 0]]
    if "s" in kind:
        cols += [vertex[:, pi, 1], vertex[:, pj, 1]]
    if "r" in kind:
        cols += [edge[:, pi, pj]]
    cols.append(np.ones((vertex.shape[0], pi.size), dtype=vertex.dtype))
    return np.stack(cols, axis=2)


def fit_pairwise(vertex, edge, y, pairs, kind, C=1.0, chunk=2048,
                 iters=50, tol=1e-9):
    """Return beta (K, q+1): coefficients then intercept, one row per pair."""
    pi = np.asarray([p[0] for p in pairs])
    pj = np.asarray([p[1] for p in pairs])
    k_total = pi.size
    q = n_features(kind)
    lam = 1.0 / C
    pen = np.eye(q + 1)
    pen[q, q] = 0.0                      # the intercept is not penalised
    yv = np.asarray(y, dtype=np.float64)
    beta = np.zeros((k_total, q + 1))

    for s in range(0, k_total, chunk):
        e = min(s + chunk, k_total)
        f = design(vertex, edge, pi[s:e], pj[s:e], kind)   # (n, k, q+1)
        f = np.ascontiguousarray(np.transpose(f, (1, 0, 2)))  # (k, n, q+1)
        b = np.zeros((e - s, q + 1))
        prev = np.inf
        for _ in range(iters):
            eta = np.einsum("knq,kq->kn", f, b)
            p = 1.0 / (1.0 + np.exp(-np.clip(eta, -500, 500)))
            w = np.clip(p * (1.0 - p), 1e-10, None)
            grad = np.einsum("knq,kn->kq", f, p - yv) + lam * (b @ pen)
            hess = np.einsum("kna,kn,knb->kab", f, w, f) + lam * pen
            hess[:, np.arange(q + 1), np.arange(q + 1)] += 1e-9
            step = np.linalg.solve(hess, grad[:, :, None])[:, :, 0]
            b -= step
            move = np.abs(step).max()
            if move < tol or abs(prev - move) < 1e-15:
                break
            prev = move
        beta[s:e] = b
    return beta


def pairwise_logits(vertex, edge, pairs, kind, beta, chunk=2048):
    """(n, K) edge log-odds = logit P(class 1 | pair features)."""
    pi = np.asarray([p[0] for p in pairs])
    pj = np.asarray([p[1] for p in pairs])
    out = np.empty((vertex.shape[0], pi.size))
    for s in range(0, pi.size, chunk):
        e = min(s + chunk, pi.size)
        f = design(vertex, edge, pi[s:e], pj[s:e], kind)    # (n, k, q+1)
        out[:, s:e] = np.einsum("nkq,kq->nk", f, beta[s:e])
    return out


def node_means(values, pairs, d):
    """Mean over incident edges, per node, for an (n, K) edge array."""
    iu0 = np.asarray([p[0] for p in pairs])
    iu1 = np.asarray([p[1] for p in pairs])
    out = np.empty((values.shape[0], d))
    for k in range(values.shape[0]):
        row = values[k]
        out[k] = (np.bincount(iu0, weights=row, minlength=d)
                  + np.bincount(iu1, weights=row, minlength=d)) / (d - 1)
    return out


def two_way(L, pairs, d):
    """L (n, K) -> node effects alpha (n, d) and interaction residual E (n, K)."""
    iu0 = np.asarray([p[0] for p in pairs])
    iu1 = np.asarray([p[1] for p in pairs])
    lbar = node_means(L, pairs, d)
    mu = L.mean(axis=1)
    alpha = (d - 1) / (d - 2) * (lbar - mu[:, None])
    E = L - mu[:, None] - alpha[:, iu0] - alpha[:, iu1]
    return lbar, alpha, E
