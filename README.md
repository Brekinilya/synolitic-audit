# synolitic-audit

What carries the signal in supervised pairwise graphs ("synolitic" graphs),
what the graph itself adds on top of simpler models, and where the construction
stops working. Evaluated on HCP task fMRI, on ABIDE resting state, on public
biomedical tables with more features than samples, and on synthetic controls
built so that the answer is known in advance.

The edge weight of such a graph is `P(class 2) - P(class 1)`, so
`2 * artanh(w)` is exactly the log-odds of the pairwise classifier. Most of
what follows is a consequence of reading the edges that way.

## Layout

- `scripts/` — the experiments, one file per question.
- `results/` — the JSON written by each run; these are the numbers quoted in
  the write-up, not a re-derivation of them.

Feature caches (`*.npy`, about 2.4 GB in total) are deliberately not committed.
They are rebuilt deterministically from the raw data by
`scripts/build_cache.py` and `scripts/e_prep.py`.

## Environments

Two, because the ABIDE path pulls in `nilearn` and the HCP path pulls in
`torch`; keeping them apart avoids forcing either upgrade on the other.

**HCP, synthetic controls, biomedical tables**

```
python 3.12.3   numpy 2.5.1   scipy 1.18.0   scikit-learn 1.9.0
torch 2.12.1+cpu   torch-geometric 2.8.0   pandas 3.0.5   networkx 3.6.1
```

**ABIDE**

```
python 3.12.3   numpy 2.5.4   scipy 1.18.1   scikit-learn 1.9.1
nilearn 0.14.1   pandas 3.0.6
```

Scripts are run as `python -I -W ignore scripts/<file>.py`. The `-I` flag
isolates the interpreter so that imports cannot pick anything up from a data
directory.

## Data

| source | contents | how it is obtained |
| --- | --- | --- |
| HCP, yeo17 parcellation | 34 regions, 7 paradigms, 830 subjects | parcellated time series, local copy |
| HCP, mmp parcellation | 379 regions, same subjects | parcellated time series, local copy |
| Published ensemble graphs | 1162 dense 379x379 graphs, Emotion contrast | from the published bundle |
| Transformer setup | pooled hidden states, N = 10000, d = 64 | produced by the upstream pipeline |
| ABIDE-I | C-PAC, filt_noglobal, CC200, 871 subjects | downloaded by `nilearn.datasets.fetch_abide_pcp` |
| Biomedical tables | leukemia, arcene, colon, DLBCL, CNS | downloaded from OpenML by `scripts/e25_biomed.py` |

HCP is access-controlled and is not redistributed here. Paths to the local
copies are currently hard-coded at the top of the scripts that need them;
pointing them at another machine means editing those constants, which is the
first thing worth moving into a config file.

## Seeds and determinism

- Splits: `GroupKFold` (no shuffling, deterministic),
  `RepeatedStratifiedKFold(random_state=0)`, `StratifiedKFold(random_state=0)`.
- Subject subsamples and region subsets: `np.random.default_rng` with an
  explicit seed; the ladders take it from `--seed`, default 0.
- Synthetic data: `default_rng(1000 * rep + n)`, so repetition `rep` at a given
  `n` reproduces exactly.
- Pairwise classifiers: `LogisticRegression(random_state=0)` from scikit-learn,
  and a deterministic batched Newton solver in `fastpair.py`.
- Bootstrap: `default_rng(seed=0)`, 10000 resamples.

Not pinned: the BLAS thread count, so the last digits of an AUC can move at the
1e-6 level. No conclusion depends on that.

## Map of the scripts

**Infrastructure**

- `fastpair.py` — batched pairwise logistic regression (damped Newton, chunked
  over pairs). Agrees with scikit-learn to 0.7 per cent on the coefficients at
  roughly six times the speed; checked by `verify_fastpair.py`.
- `build_cache.py` — per-region means, standard deviations and correlations
  written to disk as float32.
- `e_prep.py` — logit transform and two-way decomposition of the published
  graphs.
- `agg.py`, `summarize.py` — aggregation and bootstrap intervals.
- `inspect_artifacts.py` — shapes and split codes of the upstream artifacts.

**Theory and controls**

- `e5_setup1.py` — the span result on the Transformer setup: regressing the
  features onto the node channel.
- `e9_synth.py` — synthetic control whose unary and node channels are null by
  construction; logistic edges against QDA edges.
- `e11_baselines.py` — high-dimensional discriminant baselines: Ledoit-Wolf
  QDA, diagonal QDA, RDA, L1 on pairwise products, and an MLP.
- `e12_controls.py` — dense random linear embedding of the same width.
- `e23_sparse_pairs.py` — the opposite regime, where the signal lives in a few
  disjoint pairs; both regimes share this script.
- `e24_structured_control.py` — random maps matched to the pairwise shape
  (linear and quadratic per pair), plus the decay of the fitted edge slopes.

**HCP**

- `e_channels.py` — channels of the published graphs.
- `e6_rawpath.py` — the raw path with out-of-fold edges: three edge families
  and the leakage contrast.
- `e7_ladder.py`, `e8_mmp.py` — learning curves in n and the ladder in d.
- `e10_qda.py` — QDA edges on real data.
- `e13_evoked.py` — removal of the block-locked response within a record.
- `e14_blockcrop.py` — block-aligned cropping against cropping the
  concatenated series.
- `e17_decompose.py` — node summaries split into their unary and pairwise
  parts.
- `e22_gambling_null.py` — whether the GAMBLING connectivity jump is denoising
  or leakage.

**ABIDE**

- `e15_abide.py` — channels under two splitting schemes, by site and by
  subject.
- `e16_abide_protocol.py`, `e18_primary.py` — the published protocol (10x5
  cross-validation, Nadeau-Bengio correction), with and without residualisation.
- `e19b_roicurve.py` — accuracy against the number of regions, under two
  orderings. `e19_roicurve.py` is the earlier, memory-hungry version.
- `e20_site.py` — head motion against site identity.
- `e21_motion_channel.py` — the moments channel before and after motion
  removal.

**Tables with p above n**

- `e25_biomed.py` — five public tables, features screened inside the training
  fold before the graph is built, competitors given both the full and the
  screened feature set.

## Status

The numbers in `results/` are the ones used in the write-up. Two things are
open: no second person has re-run the key figures independently, and the
positive synthetic regime still needs comparison against sparse interaction
models (L1 on pairwise monomials, hierarchical lasso, interaction screening,
sparse QDA) rather than against dense-covariance discriminant analysis alone.
