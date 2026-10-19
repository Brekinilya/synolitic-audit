"""E16: the readout comparison inside the published ABIDE protocol.

Protocol as reported: ABIDE-I, 871 subjects, C-PAC filt_noglobal, CC200,
10 x 5 stratified cross-validation, residualisation on age, sex and mean FD,
and the Nadeau-Bengio corrected resampled t-test for every comparison.

Confounds are regressed out with coefficients fitted on the training fold only
and applied to both folds, so the correction cannot leak the test labels.

Readouts compared, all on the same graphs:
    wbar      mean incident edge weight, the published readout
    flat      every edge, no node aggregation
    pair      edges built from rho alone, both readouts
and, as references, logistic regression on the regional moments, on the
correlation matrix, and on both.
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
    """Remove the confound fit estimated on the training fold."""
    lin = LinearRegression().fit(ctr, xtr)
    return xtr - lin.predict(ctr), xte - lin.predict(cte)


def nadeau_bengio(diffs, n_test, n_train, alpha=0.05):
    """Corrected resampled t-test for repeated cross-validation."""
    d = np.asarray(diffs, dtype=float)
    j = len(d)
    var = d.var(ddof=1)
    corr = (1.0 / j + n_test / n_train) * var
    se = np.sqrt(max(corr, 1e-300))
    tcrit = stats.t.ppf(1 - alpha / 2, j - 1)
    t = d.mean() / se if se > 0 else 0.0
    return dict(mean=float(d.mean()), lo=float(d.mean() - tcrit * se),
                hi=float(d.mean() + tcrit * se), t=float(t),
                p=float(2 * stats.t.sf(abs(t), j - 1)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", default="results_abide_protocol.json")
    args = ap.parse_args()

    from nilearn.datasets import fetch_abide_pcp
    from fastpair import fit_pairwise, node_means, pairwise_logits

    ab = None
    for attempt in range(20):
        try:
            ab = fetch_abide_pcp(data_dir=str(DATA), pipeline="cpac",
                                 band_pass_filtering=True,
                                 global_signal_regression=False,
                                 derivatives=["rois_cc200"],
                                 quality_checked=True, verbose=0)
            break
        except Exception as exc:
            print(f"  fetch attempt {attempt}: {type(exc).__name__}", flush=True)
            time.sleep(5)
    series, pheno = ab["rois_cc200"], ab["phenotypic"]
    y = np.asarray([0 if int(g) == 2 else 1 for g in pheno["DX_GROUP"]])

    def column(name, default=0.0):
        v = np.asarray([float(x) if str(x) not in ("nan", "", "-9999") else np.nan
                        for x in pheno[name]], dtype=float)
        return np.nan_to_num(v, nan=float(np.nanmedian(v)) if np.isfinite(
            np.nanmedian(v)) else default)

    conf = np.column_stack([column("AGE_AT_SCAN"), column("SEX"),
                            column("func_mean_fd")])
    n = len(series)
    d = series[0].shape[1]
    print(f"n={n} classes={np.bincount(y)} d={d} "
          f"T median={int(np.median([s.shape[0] for s in series]))} "
          f"(full length, no crop)", flush=True)

    vertex = np.empty((n, d, 2))
    edge = np.empty((n, d, d), dtype=np.float32)
    for k, s in enumerate(series):
        z = np.asarray(s, dtype=np.float64).T
        vertex[k, :, 0] = z.mean(axis=1)
        vertex[k, :, 1] = z.std(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            edge[k] = np.nan_to_num(np.corrcoef(z), nan=0.0)
    iu0, iu1 = np.triu_indices(d, k=1)
    pairs = [(int(a), int(b)) for a, b in zip(iu0, iu1)]
    raw_all = np.concatenate([vertex[:, :, 0], vertex[:, :, 1]], axis=1)
    fc_all = np.asarray(edge[:, iu0, iu1], dtype=np.float64)
    print(f"P={len(pairs)}  features built", flush=True)

    cv = RepeatedStratifiedKFold(n_splits=args.folds, n_repeats=args.repeats,
                                 random_state=0)
    per_fold: dict[str, list] = {}
    n_test = n_train = 0
    t0 = time.time()
    for split, (tr, te) in enumerate(cv.split(raw_all, y)):
        n_test, n_train = len(te), len(tr)
        raw_tr, raw_te = residualise(raw_all[tr], raw_all[te], conf[tr], conf[te])
        fc_tr, fc_te = residualise(fc_all[tr], fc_all[te], conf[tr], conf[te])

        # Rebuild the inputs the graph construction consumes, residualised.
        v_res = np.empty_like(vertex)
        v_res[tr, :, 0], v_res[te, :, 0] = raw_tr[:, :d], raw_te[:, :d]
        v_res[tr, :, 1], v_res[te, :, 1] = raw_tr[:, d:], raw_te[:, d:]
        e_res = np.zeros((n, d, d))
        full_fc = np.empty((n, len(pairs)))
        full_fc[tr], full_fc[te] = fc_tr, fc_te
        e_res[:, iu0, iu1] = full_fc
        e_res[:, iu1, iu0] = full_fc

        out = {}
        out["LR moments"] = fit_predict(raw_tr, y[tr], raw_te)
        out["LR correlations"] = fit_predict(fc_tr, y[tr], fc_te)
        out["LR moments+corr"] = fit_predict(np.hstack([raw_tr, fc_tr]), y[tr],
                                             np.hstack([raw_te, fc_te]))
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
        if split % 5 == 0:
            print(f"  split {split+1}/{args.repeats*args.folds} "
                  f"[{time.time()-t0:.0f}s]", flush=True)

    print(f"\n{'readout':>18}   {'AUC':>14}   {'balanced acc':>14}")
    summary = {}
    for key, vals in per_fold.items():
        a = np.array(vals)
        summary[key] = dict(auc=float(a[:, 0].mean()), auc_sd=float(a[:, 0].std()),
                            bacc=float(a[:, 1].mean()),
                            bacc_sd=float(a[:, 1].std()))
        print(f"{key:>18}   {a[:,0].mean():.4f} ({a[:,0].std():.3f})   "
              f"{100*a[:,1].mean():.1f}% ({100*a[:,1].std():.1f})")

    contrasts = [("SG full: flat", "SG full: wbar"),
                 ("SG pair: flat", "SG full: wbar"),
                 ("SG pair: flat", "SG full: flat"),
                 ("LR correlations", "SG full: wbar"),
                 ("LR moments+corr", "SG pair: flat"),
                 ("SG full: wbar", "LR moments")]
    print(f"\nNadeau-Bengio corrected (J={args.repeats*args.folds}, "
          f"n_test/n_train={n_test/n_train:.3f})")
    tests = {}
    for a_key, b_key in contrasts:
        for metric, idx in (("AUC", 0), ("bacc", 1)):
            diffs = [x[idx] - z[idx] for x, z in zip(per_fold[a_key],
                                                     per_fold[b_key])]
            res = nadeau_bengio(diffs, n_test, n_train)
            tests[f"{a_key} - {b_key} [{metric}]"] = res
            star = "*" if res["p"] < 0.05 else " "
            print(f"  {a_key:>16} - {b_key:<16} {metric:>4}: "
                  f"{res['mean']:+.4f} [{res['lo']:+.4f},{res['hi']:+.4f}] "
                  f"p={res['p']:.4f}{star}")

    (CACHE / args.out).write_text(json.dumps(
        dict(n=n, d=d, repeats=args.repeats, folds=args.folds,
             summary=summary, tests=tests), indent=2))
    print(f"\nwrote {CACHE / args.out}")


if __name__ == "__main__":
    main()
