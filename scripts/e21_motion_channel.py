"""E21: the moments channel before and after removing head motion.

The published primary analysis reports that synolitic graphs do not differ from
logistic regression on the moments. If the moments channel is largely a motion
channel, that equivalence says something about head movement rather than about
the graph. This isolates the pieces: means and stds separately, each with no
confound removal, with framewise displacement removed, and with the full
age/sex/FD set removed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
OUT = Path(__file__).resolve().parent.parent / "results"


def fit_predict(xtr, ytr, xte):
    sc = StandardScaler().fit(xtr)
    clf = LogisticRegression(max_iter=5000).fit(sc.transform(xtr), ytr)
    return clf.decision_function(sc.transform(xte))


def residualise(xtr, xte, ctr, cte):
    lin = LinearRegression().fit(ctr, xtr)
    return xtr - lin.predict(ctr), xte - lin.predict(cte)


def main() -> None:
    from e18_primary import load

    ab = load()
    series, pheno = ab["rois_cc200"], ab["phenotypic"]
    y = np.asarray([0 if int(g) == 2 else 1 for g in pheno["DX_GROUP"]])

    def column(name):
        v = np.asarray([float(x) if str(x) not in ("nan", "", "-9999")
                        else np.nan for x in pheno[name]], dtype=float)
        return np.nan_to_num(v, nan=float(np.nanmedian(v)))

    age, sex, fd = column("AGE_AT_SCAN"), column("SEX"), column("func_mean_fd")
    n, d = len(series), series[0].shape[1]
    means = np.empty((n, d))
    stds = np.empty((n, d))
    for k, s in enumerate(series):
        z = np.asarray(s, dtype=np.float64).T
        means[k] = z.mean(axis=1)
        stds[k] = z.std(axis=1)

    channels = {"means": means, "stds": stds,
                "moments": np.hstack([means, stds])}
    confounds = {"none": None, "FD only": fd[:, None],
                 "age+sex+FD": np.column_stack([age, sex, fd])}

    rows: dict[str, list] = {}
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0)
    for tr, te in cv.split(means, y):
        for cname, conf in confounds.items():
            for chname, x in channels.items():
                if conf is None:
                    xtr, xte = x[tr], x[te]
                else:
                    xtr, xte = residualise(x[tr], x[te], conf[tr], conf[te])
                s = fit_predict(xtr, y[tr], xte)
                rows.setdefault(f"{chname} | {cname}", []).append(
                    (float(roc_auc_score(y[te], s)),
                     float(balanced_accuracy_score(y[te], s > 0))))
        s = fit_predict(fd[tr, None], y[tr], fd[te, None])
        rows.setdefault("FD alone | none", []).append(
            (float(roc_auc_score(y[te], s)),
             float(balanced_accuracy_score(y[te], s > 0))))

    print(f"n={n} d={d}, 10x5 stratified\n")
    print(f"{'channel | confound removed':>28}   {'AUC':>8}   {'bacc':>8}")
    summary = {}
    for key, vals in sorted(rows.items()):
        a = np.array(vals)
        summary[key] = dict(auc=float(a[:, 0].mean()), bacc=float(a[:, 1].mean()))
        print(f"{key:>28}   {a[:,0].mean():.4f}   {100*a[:,1].mean():.1f}%")

    j, ratio = 50, 0.25
    print()
    for a_key, b_key in [("stds | none", "stds | FD only"),
                         ("stds | none", "stds | age+sex+FD"),
                         ("moments | none", "moments | age+sex+FD")]:
        dif = np.array([x[1] - z[1] for x, z in zip(rows[a_key], rows[b_key])])
        se = np.sqrt((1 / j + ratio) * dif.var(ddof=1))
        p = float(2 * stats.t.sf(abs(dif.mean() / se), j - 1))
        print(f"  {a_key:>22} - {b_key:<22} bacc {dif.mean():+.4f} p={p:.4f}")
        summary[f"{a_key} - {b_key}"] = dict(delta_bacc=float(dif.mean()), p=p)

    (OUT / "abide_motion_channel.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT / 'abide_motion_channel.json'}")


if __name__ == "__main__":
    main()
