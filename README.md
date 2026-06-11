# A Comparative Study of Fingerprint Verification Algorithms

Traditional, zero-training fingerprint verification on the **FVC2002** dataset,
comparing four feature/matching approaches and several preprocessing combos.

## Algorithms compared
| Method | Feature | Matching |
|--------|---------|----------|
| SIFT   | scale-invariant keypoints | BFMatcher + Lowe ratio + RANSAC |
| ORB    | binary keypoints | BFMatcher(Hamming) + ratio + RANSAC |
| LBP    | block-based texture histogram | Chi-Square |
| Minutiae | ridge endings/bifurcations (Crossing Number) | local descriptor + RANSAC |

## Repository layout
```
core/              shared code used by both the experiments and the demo apps
  io_utils.py        image reading, dataset paths, fixed random seed
  preprocessing.py   6 combos: C1/C2/C3 (+Gabor variants)
  features.py        SIFT / ORB / LBP / Minutiae extractors
  matching.py        match + RANSAC + scoring variants S1..S4
  evaluation.py      1:1 EER, 1:N Rank-1 / identification rate
apps/              interactive demo programs (one per algorithm)
  sift_app.py        SIFT verification + 1:N attendance demo
  orb_app.py         ORB
  lbp_app.py         LBP
  minutiae_app.py    Minutiae (incl. minutiae-map visualization)
experiments/
  run_experiments.py reproducible studies (--exp 1..4)
results/           CSV outputs + figures/ + report_*.md for the report
fingerprints/      FVC2002 data (not committed — see .gitignore)
RESEARCH_LOG.md    full before/after changelog and experiment results
```

## Quick start
```bash
pip install -r requirements.txt
# place the FVC2002 set-B data under fingerprints/DB1_B .. DB4_B

# Reproducible experiments (each writes a CSV to results/ and a figure to results/figures/)
python experiments/run_experiments.py --exp 1   # preprocessing study (SIFT, 4 DBs)
python experiments/run_experiments.py --exp 2   # preprocessing generalization (4 algos)
python experiments/run_experiments.py --exp 3   # algorithm comparison: EER + latency, 1:1 & 1:N
python experiments/run_experiments.py --exp 4   # scoring strategy study (S1..S4)

# Interactive demos (menu: 1=evaluate EER, 2=1:N attendance, 3=outlier/visualize, 4=exit)
python apps/sift_app.py
```

All experiments fix a random seed (`core.io_utils.SEED = 42`) so results are reproducible.
See **RESEARCH_LOG.md** for methodology and findings, and **results/report_*.md** for
ready-to-use report text.

## Dataset
FVC2002 (set B: 10 fingers × 8 impressions per database, DB1–DB4). It is licence-restricted
and therefore not committed; download it (e.g. from Kaggle) and place each database under
`fingerprints/DB1_B`, `DB2_B`, `DB3_B`, `DB4_B`.

## License
Released under the MIT License — see [LICENSE](LICENSE).
