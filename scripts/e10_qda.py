"""E5: interaction-carrying edges on real HCP data.

The published edge learner is a logistic regression on (m_i, m_j, s_i, s_j,
rho_ij), so the edge log-odds is affine in those five numbers and the whole
graph is an affine image of them. Replacing it with a five-dimensional QDA adds
the quadratic and cross terms, so an edge can finally express an interaction.

The 2 x (5 x 5) class covariances of every pair are read off global moment
matrices, so the whole ensemble of 71631 edge models costs a handful of matrix
products instead of 71631 iterative fits.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e8_mmp import CHUNK, design_fc, fit_chunked, score  # noqa: E402

CACHE = Path(__file__).resolve().parent / "cache"


def class_moments(vertex, fc, rows, reg):
    """Per-pair 5-d mean and covariance, assembled from global moments."""
    m = np.asarray(vertex[rows, :, 0], dtype=np.float64)
    s = np.asarray(vertex[rows, :, 1], dtype=np.float64)
    f = np.asarray(fc[rows], dtype=np.float64)
    nc = m.shape[0]
    mu_m, mu_s, mu_r = m.mean(0), s.mean(0), f.mean(0)
    mc, sc, fcen = m - mu_m, s - mu_s, f - mu_r
    denom = max(nc - 1, 1)
    return dict(
        nc=nc, mu_m=mu_m, mu_s=mu_s, mu_r=mu_r,
        cmm=mc.T @ mc / denom, css=sc.T @ sc / denom, cms=mc.T @ sc / denom,
        cmr=mc.T @ fcen / denom,          # (d, P) cov(m_a, rho_k)
        csr=sc.T @ fcen / denom,          # (d, P) cov(s_a, rho_k)
        vr=(fcen * fcen).sum(0) / denom,  # (P,)
        reg=reg,
    )


def pair_gaussian(mom, pi, pj, s, e):
    """(mean (k,5), inverse covariance (k,5,5), logdet (k,)) for a pair chunk."""
    i, j = pi[s:e], pj[s:e]
    k = e - s
    mu = np.stack([mom["mu_m"][i], mom["mu_m"][j],
                   mom["mu_s"][i], mom["mu_s"][j], mom["mu_r"][s:e]], axis=1)
    cmm, css, cms = mom["cmm"], mom["css"], mom["cms"]
    cmr, csr = mom["cmr"][:, s:e], mom["csr"][:, s:e]
    cov = np.empty((k, 5, 5))
    cov[:, 0, 0] = cmm[i, i]; cov[:, 1, 1] = cmm[j, j]
    cov[:, 2, 2] = css[i, i]; cov[:, 3, 3] = css[j, j]
    cov[:, 4, 4] = mom["vr"][s:e]
    cov[:, 0, 1] = cov[:, 1, 0] = cmm[i, j]
    cov[:, 2, 3] = cov[:, 3, 2] = css[i, j]
    cov[:, 0, 2] = cov[:, 2, 0] = cms[i, i]
    cov[:, 0, 3] = cov[:, 3, 0] = cms[i, j]
    cov[:, 1, 2] = cov[:, 2, 1] = cms[j, i]
    cov[:, 1, 3] = cov[:, 3, 1] = cms[j, j]
    cov[:, 0, 4] = cov[:, 4, 0] = cmr[i, np.arange(k)]
    cov[:, 1, 4] = cov[:, 4, 1] = cmr[j, np.arange(k)]
    cov[:, 2, 4] = cov[:, 4, 2] = csr[i, np.arange(k)]
    cov[:, 3, 4] = cov[:, 4, 3] = csr[j, np.arange(k)]
    a = mom["reg"]
    if a:
        tr = np.trace(cov, axis1=1, axis2=2) / 5.0
        cov = (1 - a) * cov + a * tr[:, None, None] * np.eye(5)
    cov[:, np.arange(5), np.arange(5)] += 1e-10
    sign, logdet = np.linalg.slogdet(cov)
    return mu, np.linalg.inv(cov), logdet


def qda_edges(vertex, fc, y, rows, pi, pj, reg=0.1):
    """(n, P) edge log-odds from a 5-d QDA per pair."""
    rows = np.asarray(rows)
    mom = {c: class_moments(vertex, fc, rows[y[rows] == c], reg) for c in (0, 1)}
    prior = np.log(mom[1]["nc"] / max(mom[0]["nc"], 1))
    n = vertex.shape[0]
    out = np.empty((n, pi.size), dtype=np.float32)
    vall = np.asarray(vertex, dtype=np.float64)
    for s in range(0, pi.size, CHUNK):
        e = min(s + CHUNK, pi.size)
        fcc = np.asarray(fc[:, s:e], dtype=np.float64)
        f = design_fc(vall, fcc, pi[s:e], pj[s:e], "mmssr")[:, :, :5]
        quad = {}
        ld = {}
        for c in (0, 1):
            mu, inv, logdet = pair_gaussian(mom[c], pi, pj, s, e)
            z = f - mu[None, :, :]
            quad[c] = np.einsum("nka,kab,nkb->nk", z, inv, z)
            ld[c] = logdet
        out[:, s:e] = (-0.5 * (quad[1] - quad[0])
                       - 0.5 * (ld[1] - ld[0]) + prior).astype(np.float32)
    return out


def node_mean_chunked(L, pi, pj, d):
    out = np.zeros((L.shape[0], d))
    for s in range(0, pi.size, CHUNK):
        e = min(s + CHUNK, pi.size)
        k = e - s
        inc = sp.csr_matrix(
            (np.ones(2 * k),
             (np.concatenate([np.arange(k), np.arange(k)]),
              np.concatenate([pi[s:e], pj[s:e]]))), shape=(k, d))
        out += np.asarray(L[:, s:e], dtype=np.float64) @ inc
    return out / (d - 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default="yeo17_EMOTION_RL")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--train-sizes", default="16,32,64,128,256,0")
    ap.add_argument("--reg", type=float, default=0.1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_name = args.out or ("qda_" + args.stem + ".json")
    meta = json.loads((CACHE / (args.stem + "_meta.json")).read_text())
    vertex = np.asarray(np.load(CACHE / (args.stem + "_vertex.npy"),
                                mmap_mode="r"))
    y = np.asarray(meta["y"])
    groups = np.unique(np.asarray(meta["subject"]), return_inverse=True)[1]
    d, n = meta["d"], meta["n"]
    pi, pj = np.triu_indices(d, k=1)
    fc_path = CACHE / (args.stem + "_fc.npy")
    if not fc_path.exists():
        edge = np.load(CACHE / (args.stem + "_edge.npy"), mmap_mode="r")
        build = np.lib.format.open_memmap(fc_path, mode="w+", dtype=np.float32,
                                          shape=(n, pi.size))
        for k in range(n):
            build[k] = edge[k][pi, pj]
        build.flush()
        del edge, build
    fc = np.asarray(np.load(fc_path, mmap_mode="r"))
    raw = np.concatenate([vertex[:, :, 0], vertex[:, :, 1]], axis=1)
    print(f"{args.stem}: n={n} d={d} P={pi.size} reg={args.reg}", flush=True)

    rows = []
    for fold, (tr, te) in enumerate(
            GroupKFold(n_splits=args.folds).split(raw, y, groups)):
        tr_subj = np.unique(groups[tr])
        for size in [int(x) for x in args.train_sizes.split(",")]:
            if size and size >= len(tr_subj):
                continue
            rng = np.random.default_rng(1000 * fold + size)
            sel = (tr[np.isin(groups[tr], rng.choice(tr_subj, size=size,
                                                     replace=False))]
                   if size else tr)
            ts = time.time()
            beta = fit_chunked(vertex, fc, y, pi, pj, "mmssr", sel)
            ll = np.empty((n, pi.size), dtype=np.float32)
            for s in range(0, pi.size, CHUNK):
                e = min(s + CHUNK, pi.size)
                f = design_fc(np.asarray(vertex, dtype=np.float64),
                              np.asarray(fc[:, s:e], dtype=np.float64),
                              pi[s:e], pj[s:e], "mmssr")
                ll[:, s:e] = np.einsum("nkq,kq->nk", f, beta[s:e])
            lq = qda_edges(vertex, fc, y, sel, pi, pj, reg=args.reg)
            res = {"raw (m,s)": score(raw[sel], y[sel], raw[te], y[te]),
                   "raw + FC": score(np.hstack([raw[sel], fc[sel]]), y[sel],
                                     np.hstack([raw[te], fc[te]]), y[te])}
            for tag, L in [("LR", ll), ("QDA", lq)]:
                nm = node_mean_chunked(L, pi, pj, d)
                res[f"syn-{tag}: lbar"] = score(nm[sel], y[sel], nm[te], y[te])
                res[f"syn-{tag}: flat"] = score(L[sel], y[sel], L[te], y[te])
                pool = np.asarray(L, dtype=np.float64).sum(axis=1)
                sgn = 1.0 if roc_auc_score(y[sel], pool[sel]) >= 0.5 else -1.0
                res[f"syn-{tag}: pool"] = float(roc_auc_score(y[te],
                                                             sgn * pool[te]))
            rows.append(dict(fold=fold, n_subj=int(len(np.unique(groups[sel]))),
                             **res, secs=round(time.time() - ts, 1)))
            print(f"fold={fold} n_subj={rows[-1]['n_subj']:>4d}  "
                  + "  ".join(f"{k}={v:.4f}" for k, v in res.items())
                  + f"  [{rows[-1]['secs']}s]", flush=True)
            (CACHE / out_name).write_text(json.dumps(
                dict(stem=args.stem, d=d, n=n, reg=args.reg, rows=rows),
                indent=2))
    print("\nwrote " + str(CACHE / out_name))


if __name__ == "__main__":
    main()
