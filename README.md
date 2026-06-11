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
core/          shared code (single source of truth, no duplication)
  io_utils.py      image reading, dataset paths, random seed
  preprocessing.py 6 combos: C1/C2/C3 (+Gabor variants)
  features.py      SIFT / ORB / LBP / Minutiae extractors
  matching.py      match + RANSAC + scoring variants S1..S4
  evaluation.py    1:1 EER, 1:N Rank-1 / identification rate
Biology.py / ORB.py / lbp_main.py / Minutiae.py   interactive demo programs
               (one per algorithm; currently at repo root, will move to apps/
                once refactored to import from core/)
experiments/   run_experiments.py — reproducible studies (--exp 1..4)
results/       CSV outputs + figures/ for the report
fingerprints/  FVC2002 data (not committed — see .gitignore)
RESEARCH_LOG.md  full before/after changelog and experiment results
```

## Quick start
```bash
pip install -r requirements.txt
# place FVC2002 data under fingerprints/DB1_B .. DB4_B
python experiments/run_experiments.py --exp 1     # preprocessing study
python apps/sift_app.py                            # interactive demo
```

All experiments fix a random seed (`core.io_utils.SEED`) so results are reproducible.
See **RESEARCH_LOG.md** for methodology, before/after changes, and findings.
