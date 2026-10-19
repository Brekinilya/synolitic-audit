"""E7 at d=379: learning curves and the dimensionality ladder on mmp.

Memory-lean: functional connectivity is held once as an (n, P) float32 memmap
and the dense (n, d, d) stack is never materialised. Node summaries are
accumulated in chunks over pairs, so the (n, P) log-odds matrix is not kept
either unless --with-flat is given.

Edge models are refitted on each training subsample, so no condition sees data
the baselines do not see.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

CACHE = Path(__file__).resolve().parent / "cache"
sys.path.insert(0, str(Path(__file__).resolve().parent))
CHUNK = 2048


def score(xtr, ytr, xte, yte, C=1.0):
    if xtr.ndim == 1:
        xtr, xte = xtr[:, None], xte[:, None]
    sc = StandardScaler().fit(xtr)
    kw = dict(C=C, max_iter=5000)
    if xtr.shape[1] > 5000:
        kw.update(dual=True, solver="liblinear")
    clf = LogisticRegression(**kw).fit(sc.transform(xtr), ytr)
    return float(roc_auc_score(yte, clf.decision_function(sc.transform(xte))))


def design_fc(vertex, fcc, pi, pj, kind):
    """(n, k, q+1) design for a chunk of pairs, rho taken from fc columns."""
    cols = []
    if "m" in kind:
        cols += [vertex[:, pi, 0], vertex[:, pj, 0]]
    if "s" in kind:
        cols += [vertex[:, pi, 1], vertex[:, pj, 1]]
    if "r" in kind:
        cols += [fcc]
    cols.append(np.ones((vertex.shape[0], pi.size)))
    return np.stack([np.asarray(c, dtype=np.float64) for c in cols], axis=2)


def fit_chunked(vertex, fc, y, pi, pj, kind, rows, C=1.0, iters=40, tol=1e-9):
    q = 2 * ("m" in kind) + 2 * ("s" in kind) + 1 * ("r" in kind)
    lam, pen = 1.0 / C, np.eye(q + 1)
    pen[q, q] = 0.0
    yv = np.asarray(y[rows], dtype=np.float64)
    beta = np.zeros((pi.size, q + 1))
    vsub = np.asarray(vertex[rows], dtype=np.float64)
    for s in range(0, pi.size, CHUNK):
        e = min(s + CHUNK, pi.size)
        fcc = np.asarray(fc[rows, s:e], dtype=np.float64) if "r" in kind else None
        f = np.ascontiguousarray(
            np.transpose(design_fc(vsub, fcc, pi[s:e], pj[s:e], kind), (1, 0, 2)))
        b = np.zeros((e - s, q + 1))
        for _ in range(iters):
            p = 1.0 / (1.0 + np.exp(-np.clip(np.einsum("knq,kq->kn", f, b),
                                             -500, 500)))
            w = np.clip(p * (1.0 - p), 1e-10, None)
            g = np.einsum("knq,kn->kq", f, p - yv) + lam * (b @ pen)
            h = np.einsum("kna,kn,knb->kab", f, w, f) + lam * pen
            h[:, np.arange(q + 1), np.arange(q + 1)] += 1e-9
            step = np.linalg.solve(h, g[:, :, None])[:, :, 0]
            b -= step
            if np.abs(step).max() < tol:
                break
        beta[s:e] = b
    return beta


def readouts(vertex, fc, pi, pj, kind, beta, d, keep_flat=False):
    """Node summaries in log-odds and in w, plus the unweighted pool."""
    n = vertex.shape[0]
    lsum = np.zeros((n, d))
    wsum = np.zeros((n, d))
    pool = np.zeros(n)
    flat = np.empty((n, pi.size), dtype=np.float32) if keep_flat else None
    vall = np.asarray(vertex, dtype=np.float64)
    for s in range(0, pi.size, CHUNK):
        e = min(s + CHUNK, pi.size)
        fcc = np.asarray(fc[:, s:e], dtype=np.float64) if "r" in kind else None
        f = design_fc(vall, fcc, pi[s:e], pj[s:e], kind)      # (n, k, q+1)
        lc = np.einsum("nkq,kq->nk", f, beta[s:e])
        wc = np.tanh(lc / 2.0)
        if keep_flat:
            flat[:, s:e] = lc.astype(np.float32)
        pool += lc.sum(axis=1)
        k = e - s
        inc = sp.csr_matrix(
            (np.ones(2 * k),
             (np.concatenate([np.arange(k), np.arange(k)]),
              np.concatenate([pi[s:e], pj[s:e]]))), shape=(k, d))
        lsum += lc @ inc
        wsum += wc @ inc
    return lsum / (d - 1), wsum / (d - 1), pool, flat


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default="mmp_EMOTION_RL")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--d-ladder", default="379")
    ap.add_argument("--region-draws", type=int, default=1)
    ap.add_argument("--train-sizes", default="16,32,64,128,256,0")
    ap.add_argument("--family", default="mmssr")
    ap.add_argument("--with-flat", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_name = args.out or ("ladder_" + args.stem + ".json")
    meta = json.loads((CACHE / (args.stem + "_meta.json")).read_text())
    vertex_all = np.load(CACHE / (args.stem + "_vertex.npy"), mmap_mode="r")
    y = np.asarray(meta["y"])
    groups = np.unique(np.asarray(meta["subject"]), return_inverse=True)[1]
    d_all, n = meta["d"], meta["n"]
    print(f"{args.stem}: n={n} d={d_all} subjects={groups.max()+1} "
          f"T={meta['crop']}", flush=True)

    fc_path = CACHE / (args.stem + "_fc.npy")
    iu0_all, iu1_all = np.triu_indices(d_all, k=1)
    if not fc_path.exists():
        t0 = time.time()
        edge = np.load(CACHE / (args.stem + "_edge.npy"), mmap_mode="r")
        fc_build = np.lib.format.open_memmap(
            fc_path, mode="w+", dtype=np.float32, shape=(n, iu0_all.size))
        for k in range(n):
            fc_build[k] = edge[k][iu0_all, iu1_all]
        fc_build.flush()
        del edge, fc_build
        print(f"built fc cache [{time.time()-t0:.0f}s]", flush=True)
    fc_all = np.load(fc_path, mmap_mode="r")

    key = {(int(a), int(b)): k for k, (a, b) in enumerate(zip(iu0_all, iu1_all))}
    ladder = [int(x) for x in args.d_ladder.split(",")]
    sizes = [int(x) for x in args.train_sizes.split(",")]
    rng = np.random.default_rng(args.seed)
    rows: list[dict] = []

    for d in ladder:
        for draw in range(args.region_draws if d < d_all else 1):
            reg = (np.sort(rng.choice(d_all, size=d, replace=False))
                   if d < d_all else np.arange(d_all))
            pi, pj = np.triu_indices(d, k=1)
            cols = np.array([key[(int(reg[a]), int(reg[b]))]
                             for a, b in zip(pi, pj)])
            vertex = np.ascontiguousarray(vertex_all[:, reg])
            fc = np.ascontiguousarray(fc_all[:, cols]) if d < d_all else fc_all
            raw = np.concatenate([vertex[:, :, 0], vertex[:, :, 1]], axis=1)

            for fold, (tr, te) in enumerate(
                    GroupKFold(n_splits=args.folds).split(raw, y, groups)):
                tr_subj = np.unique(groups[tr])
                for size in sizes:
                    if size and size >= len(tr_subj):
                        continue
                    sel = (tr[np.isin(groups[tr],
                                      rng.choice(tr_subj, size=size,
                                                 replace=False))]
                           if size else tr)
                    ts = time.time()
                    beta = fit_chunked(vertex, fc, y, pi, pj, args.family, sel)
                    lbar, wbar, pool, flat = readouts(
                        vertex, fc, pi, pj, args.family, beta, d,
                        keep_flat=args.with_flat)
                    sgn = 1.0 if roc_auc_score(y[sel], pool[sel]) >= 0.5 else -1.0
                    fc_tr = np.asarray(fc[sel])
                    fc_te = np.asarray(fc[te])
                    res = {
                        "raw (m,s)": score(raw[sel], y[sel], raw[te], y[te]),
                        "FC vec(rho)": score(fc_tr, y[sel], fc_te, y[te]),
                        "raw + FC": score(np.hstack([raw[sel], fc_tr]), y[sel],
                                          np.hstack([raw[te], fc_te]), y[te]),
                        "syn: wbar": score(wbar[sel], y[sel], wbar[te], y[te]),
                        "syn: lbar": score(lbar[sel], y[sel], lbar[te], y[te]),
                        "syn: sum_l": float(roc_auc_score(y[te], sgn * pool[te])),
                    }
                    if args.with_flat:
                        res["syn: Lflat"] = score(flat[sel], y[sel],
                                                  flat[te], y[te])
                    del fc_tr, fc_te, flat
                    rows.append(dict(d=d, draw=draw, fold=fold,
                                     n_subj=int(len(np.unique(groups[sel]))),
                                     n_rec=int(len(sel)), **res,
                                     secs=round(time.time() - ts, 1)))
                    print(f"d={d:>4d} draw={draw} fold={fold} "
                          f"n_subj={rows[-1]['n_subj']:>4d}  "
                          + "  ".join(f"{k}={v:.4f}" for k, v in res.items())
                          + f"  [{rows[-1]['secs']}s]", flush=True)
                    (CACHE / out_name).write_text(
                        json.dumps(dict(parcel="mmp", task=meta["task"],
                                        direction=meta["direction"],
                                        d_all=d_all, n=n, crop=meta["crop"],
                                        family=args.family, rows=rows), indent=2))
    print("\nwrote " + str(CACHE / out_name))


if __name__ == "__main__":
    main()
