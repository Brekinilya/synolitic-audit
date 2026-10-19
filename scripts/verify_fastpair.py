"""Check the batched pairwise fitter against sklearn on real yeo17 features."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

REPO = Path(r"C:\Users\breki\smiles-synolitic-correction")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(r"C:\Users\breki\Downloads\HCP_v4\yeo17_extracted\split_yeo17_lr")


def main() -> None:
    from fastpair import fit_pairwise, pairwise_logits
    from synolitic.setup2.data import build_task_features, scan_records

    recs = [r for r in scan_records(ROOT, "EMOTION") if r.direction == "RL"][:600]
    tf = build_task_features(recs)
    vertex, edge, y = tf.vertex, tf.edge, tf.y
    d = vertex.shape[1]
    pairs = [(i, j) for i in range(d) for j in range(i + 1, d)]

    for kind in ["mmssr", "mmss", "r"]:
        t0 = time.time()
        beta = fit_pairwise(vertex, edge, y, pairs, kind)
        t_fast = time.time() - t0

        t0 = time.time()
        ref = []
        for i, j in pairs[:120]:
            cols = []
            if "m" in kind:
                cols += [vertex[:, i, 0], vertex[:, j, 0]]
            if "s" in kind:
                cols += [vertex[:, i, 1], vertex[:, j, 1]]
            if "r" in kind:
                cols += [edge[:, i, j]]
            f = np.column_stack(cols)
            clf = LogisticRegression(random_state=0, max_iter=1000).fit(f, y)
            ref.append(np.append(clf.coef_[0], clf.intercept_[0]))
        t_ref = time.time() - t0
        ref = np.asarray(ref)
        diff = np.abs(ref - beta[:120])
        scale = np.maximum(np.abs(ref).max(), 1e-12)
        print(f"kind={kind:>6s}  max|dbeta|={diff.max():.3e}  "
              f"rel={diff.max()/scale:.3e}  "
              f"batched {t_fast:.2f}s for {len(pairs)} pairs vs "
              f"sklearn {t_ref:.2f}s for 120")

    L = pairwise_logits(vertex, edge, pairs, "mmssr",
                        fit_pairwise(vertex, edge, y, pairs, "mmssr"))
    w = np.tanh(L / 2.0)
    print(f"\nlogit->w check: w range [{w.min():.4f}, {w.max():.4f}], "
          f"L shape {L.shape}")


if __name__ == "__main__":
    main()
