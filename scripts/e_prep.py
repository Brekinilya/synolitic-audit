"""E1/E2 preparation: logit transform and two-way decomposition of HCP ensemble graphs.

Reads the published Vlasenko Emotion ensemble graphs (1162 x 379 x 379 signed
weights w = P(c2) - P(c1)) and writes, to the scratchpad cache:

    W.npy      (N, P) float32   upper-triangular signed weights
    L.npy      (N, P) float32   edge log-odds, L = 2*artanh(w) = logit p
    E.npy      (N, P) float32   two-way residual L_ij - mu - alpha_i - alpha_j
    nodes.npz                   node-level channels + labels + subjects

Two-way decomposition (complete graph, no diagonal, sum alpha = 0):
    mu      = mean_{i<j} L_ij
    abar_i  = mean_{j != i} L_ij
    alpha_i = (d-1)/(d-2) * (abar_i - mu)
    e_ij    = L_ij - mu - alpha_i - alpha_j      (row sums exactly zero)
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(r"C:\Users\breki\smiles-synolitic-correction")
CACHE = Path(__file__).resolve().parent / "cache"
EPS = 1e-6
CHUNK = 128


def main() -> None:
    CACHE.mkdir(exist_ok=True)
    t0 = time.time()

    obj = torch.load(REPO / "artifacts/setup2/emotion_ensemble_test.pt",
                     map_location="cpu", weights_only=False)
    adj = obj["adj"].numpy()
    y = obj["y"].numpy().astype(np.int64)
    subject = np.asarray(obj["subject"])
    n, d, _ = adj.shape
    print(f"loaded adj {adj.shape} y {np.bincount(y)} "
          f"subjects {len(np.unique(subject))}  [{time.time()-t0:.1f}s]")

    # Node-level channels straight from the dense matrices (cheap, one pass).
    denom = d - 1
    wbar = adj.sum(axis=2) / denom                      # signed mean incident w
    absbar = np.abs(adj).sum(axis=2) / denom            # magnitude-only channel
    print(f"wbar {wbar.shape}  |w| range {np.abs(adj).min():.4g}..{np.abs(adj).max():.4g}")

    iu0, iu1 = np.triu_indices(d, k=1)
    p = iu0.size
    print(f"pairs P = {p}")

    W = np.empty((n, p), dtype=np.float32)
    for s in range(0, n, CHUNK):
        W[s:s + CHUNK] = adj[s:s + CHUNK][:, iu0, iu1]
    del adj, obj
    print(f"W built {W.shape}  [{time.time()-t0:.1f}s]")

    # Edge log-odds: 2*artanh(w) == logit(p) exactly.
    L = np.empty_like(W)
    for s in range(0, n, CHUNK):
        chunk = np.clip(W[s:s + CHUNK].astype(np.float64), -1 + EPS, 1 - EPS)
        L[s:s + CHUNK] = (2.0 * np.arctanh(chunk)).astype(np.float32)
    sat = float(np.mean(np.abs(W) > 1 - 1e-3))
    print(f"L built, |w|>1-1e-3 fraction = {sat:.4%}, "
          f"|L| max = {np.abs(L).max():.3f}  [{time.time()-t0:.1f}s]")

    # Two-way (additive) decomposition of the symmetric edge matrix.
    mu = L.mean(axis=1, dtype=np.float64)                       # (n,)
    lsum = np.zeros((n, d), dtype=np.float64)
    for k in range(n):
        row = L[k].astype(np.float64)
        lsum[k] = (np.bincount(iu0, weights=row, minlength=d)
                   + np.bincount(iu1, weights=row, minlength=d))
    lbar = lsum / denom                                          # abar_i
    alpha = (d - 1) / (d - 2) * (lbar - mu[:, None])

    E = np.empty_like(L)
    for s in range(0, n, CHUNK):
        a = alpha[s:s + CHUNK]
        E[s:s + CHUNK] = (L[s:s + CHUNK].astype(np.float64)
                          - mu[s:s + CHUNK, None]
                          - a[:, iu0] - a[:, iu1]).astype(np.float32)
    # Sanity: residual row sums must vanish.
    row0 = E[0].astype(np.float64)
    rs = (np.bincount(iu0, weights=row0, minlength=d)
          + np.bincount(iu1, weights=row0, minlength=d))
    print(f"residual row-sum max |.| (graph 0) = {np.abs(rs).max():.3e}  "
          f"(scale |L| ~ {np.abs(L[0]).mean():.3f})")
    print(f"E built  [{time.time()-t0:.1f}s]")

    np.save(CACHE / "W.npy", W)
    np.save(CACHE / "L.npy", L)
    np.save(CACHE / "E.npy", E)
    np.savez(CACHE / "nodes.npz", y=y, subject=subject, wbar=wbar.astype(np.float32),
             absbar=absbar.astype(np.float32), lbar=lbar.astype(np.float32),
             alpha=alpha.astype(np.float32), mu=mu.astype(np.float32),
             wsum=(wbar.sum(axis=1) * denom / 2).astype(np.float32),
             lsum=(lbar.sum(axis=1) * denom / 2).astype(np.float32),
             d=d, n=n, p=p)
    print(f"cached to {CACHE}  [{time.time()-t0:.1f}s]")


if __name__ == "__main__":
    main()
