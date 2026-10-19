"""E18: the primary ABIDE analysis, with and without residualisation.

Fixes a comparison error: the published 57.0 / 67.9 come from the primary
analysis, while 55.6 / 66.9-67.1 are the residualised ones, and the reported
"synolitic graphs do not differ from logistic regression on the moments"
(p = 0.82) is a primary-analysis statement. Both conditions are run here so the
comparison is like for like.

Extra rows that the mechanism argument needs:

  means / stds        the moments channel split, because temporal std is the
                      part most entangled with head motion
  corr graph strength an unsupervised correlation graph read by node summaries,
                      which isolates the readout from the supervision
  rho node effects    the two-way fit rho_ij = mu + alpha_i + alpha_j + e_ij,
  rho residual        where node averaging keeps alpha exactly and annihilates
                      e exactly, e being every edge pattern whose node sums
                      vanish

The last pair replaces the vague "shrinkage" numbers with the quantity the
mechanism is actually about: what node averaging keeps versus what it destroys.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
CACHE = Path(__file__).resolve().parent / "cache"
DATA = Path("C:/Users/breki/abide_data")


def fit_predict(xtr, ytr, xte, C=1.0):
    sc = StandardScaler().fit(xtr)
    kw = dict(C=C, max_iter=5000)
    if xtr.shape[1] > 5000:
        kw.update(dual=True, solver="liblinear")
    clf = LogisticRegression(**kw).fit(sc.transform(xtr), ytr)
    return clf.decision_function(sc.transform(xte))


def residualise(xtr, xte, ctr, cte):
    lin = LinearRegression().fit(ctr, xtr)
    return xtr - lin.predict(ctr), xte - lin.predict(cte)


def nadeau_bengio(diffs, n_test, n_train, alpha=0.05):
    d = np.asarray(diffs, dtype=float)
    j = len(d)
    se = np.sqrt(max((1.0 / j + n_test / n_train) * d.var(ddof=1), 1e-300))
    tcrit = stats.t.ppf(1 - alpha / 2, j - 1)
    t = d.mean() / se if se > 0 else 0.0
    return dict(mean=float(d.mean()), lo=float(d.mean() - tcrit * se),
                hi=float(d.mean() + tcrit * se), t=float(t),
                p=float(2 * stats.t.sf(abs(t), j - 1)))


def two_way(vec, pi, pj, d):
    """rho_ij = mu + alpha_i + alpha_j + e_ij, with e having zero node sums."""
    n = vec.shape[0]
    mu = vec.mean(axis=1)
    bar = np.empty((n, d))
    for k in range(n):
        bar[k] = (np.bincount(pi, weights=vec[k], minlength=d)
                  + np.bincount(pj, weights=vec[k], minlength=d)) / (d - 1)
    alpha = (d - 1) / (d - 2) * (bar - mu[:, None])
    resid = vec - mu[:, None] - alpha[:, pi] - alpha[:, pj]
    return alpha, resid


def load():
    from nilearn.datasets import fetch_abide_pcp
    ab = None
    for _ in range(20):
        try:
            ab = fetch_abide_pcp(data_dir=str(DATA), pipeline="cpac",
                                 band_pass_filtering=True,
                                 global_signal_regression=False,
                                 derivatives=["rois_cc200"],
                                 quality_checked=True, verbose=0)
            break
        except Exception:
            time.sleep(5)
    return ab


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--residualise", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out_name = args.out or f"abide_primary_resid{args.residualise}.json"

    from fastpair import fit_pairwise, node_means, pairwise_logits

    ab = load()
    series, pheno = ab["rois_cc200"], ab["phenotypic"]
    y = np.asarray([0 if int(g) == 2 else 1 for g in pheno["DX_GROUP"]])

    def column(name):
        v = np.asarray([float(x) if str(x) not in ("nan", "", "-9999")
                        else np.nan for x in pheno[name]], dtype=float)
        return np.nan_to_num(v, nan=float(np.nanmedian(v)))

    age, sex, fd = column("AGE_AT_SCAN"), column("SEX"), column("func_mean_fd")
    conf = np.column_stack([age, sex, fd])
    n, d = len(series), series[0].shape[1]

    vertex = np.empty((n, d, 2))
    edge = np.empty((n, d, d), dtype=np.float32)
    for k, s in enumerate(series):
        z = np.asarray(s, dtype=np.float64).T
        vertex[k, :, 0] = z.mean(axis=1)
        vertex[k, :, 1] = z.std(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            edge[k] = np.nan_to_num(np.corrcoef(z), nan=0.0)
    pi, pj = np.triu_indices(d, k=1)
    pairs = [(int(a), int(b)) for a, b in zip(pi, pj)]
    means_all = vertex[:, :, 0].copy()
    stds_all = vertex[:, :, 1].copy()
    fc_all = np.asarray(edge[:, pi, pj], dtype=np.float64)

    sigma_bar = stds_all.mean(axis=1)
    r_fd = stats.pearsonr(sigma_bar, fd)
    print(f"n={n} d={d} residualise={bool(args.residualise)}")
    print(f"motion: corr(mean sigma, FD) = {r_fd[0]:+.3f} (p={r_fd[1]:.2e});"
          f"  AUC of FD alone = {roc_auc_score(y, fd):.4f};"
          f"  AUC of mean sigma = {roc_auc_score(y, sigma_bar):.4f}", flush=True)

    cv = RepeatedStratifiedKFold(n_splits=args.folds, n_repeats=args.repeats,
                                 random_state=0)
    per_fold: dict[str, list] = {}
    n_test = n_train = 0
    t0 = time.time()
    for split, (tr, te) in enumerate(cv.split(means_all, y)):
        n_test, n_train = len(te), len(tr)
        if args.residualise:
            m_tr, m_te = residualise(means_all[tr], means_all[te], conf[tr], conf[te])
            s_tr, s_te = residualise(stds_all[tr], stds_all[te], conf[tr], conf[te])
            f_tr, f_te = residualise(fc_all[tr], fc_all[te], conf[tr], conf[te])
        else:
            m_tr, m_te = means_all[tr], means_all[te]
            s_tr, s_te = stds_all[tr], stds_all[te]
            f_tr, f_te = fc_all[tr], fc_all[te]
        raw_tr, raw_te = np.hstack([m_tr, s_tr]), np.hstack([m_te, s_te])

        v_res = np.empty_like(vertex)
        v_res[tr, :, 0], v_res[te, :, 0] = m_tr, m_te
        v_res[tr, :, 1], v_res[te, :, 1] = s_tr, s_te
        full_fc = np.empty((n, len(pairs)))
        full_fc[tr], full_fc[te] = f_tr, f_te
        e_res = np.zeros((n, d, d))
        e_res[:, pi, pj] = full_fc
        e_res[:, pj, pi] = full_fc

        out = {
            "LR means": fit_predict(m_tr, y[tr], m_te),
            "LR stds": fit_predict(s_tr, y[tr], s_te),
            "LR moments": fit_predict(raw_tr, y[tr], raw_te),
            "LR correlations": fit_predict(f_tr, y[tr], f_te),
            "LR moments+corr": fit_predict(np.hstack([raw_tr, f_tr]), y[tr],
                                           np.hstack([raw_te, f_te])),
        }
        strength = np.empty((n, d))
        for k in range(n):
            strength[k] = (np.bincount(pi, weights=full_fc[k], minlength=d)
                           + np.bincount(pj, weights=full_fc[k],
                                         minlength=d)) / (d - 1)
        out["Corr graph: strength"] = fit_predict(strength[tr], y[tr],
                                                  strength[te])
        alpha, resid = two_way(full_fc, pi, pj, d)
        out["rho: node effects"] = fit_predict(alpha[tr], y[tr], alpha[te])
        out["rho: residual"] = fit_predict(resid[tr], y[tr], resid[te])
        for fam, kind in [("SG full", "mmssr"), ("SG pair", "r")]:
            beta = fit_pairwise(v_res[tr], e_res[tr], y[tr], pairs, kind)
            L = pairwise_logits(v_res, e_res, pairs, kind, beta)
            wbar = node_means(np.tanh(L / 2), pairs, d)
            out[f"{fam}: wbar"] = fit_predict(wbar[tr], y[tr], wbar[te])
            out[f"{fam}: flat"] = fit_predict(L[tr], y[tr], L[te])
        for key, s in out.items():
            per_fold.setdefault(key, []).append(
                (float(roc_auc_score(y[te], s)),
                 float(balanced_accuracy_score(y[te], s > 0))))
        if split % 10 == 0:
            print(f"  split {split+1}/{args.repeats*args.folds} "
                  f"[{time.time()-t0:.0f}s]", flush=True)

    print(f"\n{'readout':>22}   {'AUC':>14}   {'balanced acc':>14}")
    summary = {}
    for key, vals in per_fold.items():
        a = np.array(vals)
        summary[key] = dict(auc=float(a[:, 0].mean()), bacc=float(a[:, 1].mean()),
                            auc_sd=float(a[:, 0].std()),
                            bacc_sd=float(a[:, 1].std()))
        print(f"{key:>22}   {a[:,0].mean():.4f} ({a[:,0].std():.3f})   "
              f"{100*a[:,1].mean():.1f}% ({100*a[:,1].std():.1f})")

    contrasts = [("SG full: wbar", "LR moments"),
                 ("SG full: wbar", "Corr graph: strength"),
                 ("SG full: flat", "SG full: wbar"),
                 ("SG pair: flat", "SG full: flat"),
                 ("LR correlations", "SG full: wbar"),
                 ("LR correlations", "rho: node effects"),
                 ("rho: residual", "rho: node effects")]
    print(f"\nNadeau-Bengio (J={args.repeats*args.folds}, "
          f"ratio={n_test/n_train:.3f})")
    tests = {}
    for a_key, b_key in contrasts:
        for metric, idx in (("AUC", 0), ("bacc", 1)):
            diffs = [x[idx] - z[idx] for x, z in zip(per_fold[a_key],
                                                     per_fold[b_key])]
            res = nadeau_bengio(diffs, n_test, n_train)
            tests[f"{a_key} - {b_key} [{metric}]"] = res
            star = "*" if res["p"] < 0.05 else " "
            print(f"  {a_key:>20} - {b_key:<20} {metric:>4}: {res['mean']:+.4f} "
                  f"[{res['lo']:+.4f},{res['hi']:+.4f}] p={res['p']:.4f}{star}")

    (CACHE / out_name).write_text(json.dumps(
        dict(n=n, d=d, residualise=bool(args.residualise),
             repeats=args.repeats, folds=args.folds,
             motion=dict(corr_sigma_fd=float(r_fd[0]),
                         auc_fd=float(roc_auc_score(y, fd)),
                         auc_sigma=float(roc_auc_score(y, sigma_bar))),
             summary=summary, tests=tests), indent=2))
    print(f"\nwrote {CACHE / out_name}")


if __name__ == "__main__":
    main()
