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

# Feature-extraction showcase (SIFT/ORB/LBP/Minutiae features drawn on a fingerprint)
python experiments/visualize_extraction.py DB1_B/101_1.tif
```

All experiments fix a random seed (`core.io_utils.SEED = 42`) so results are reproducible.
See **RESEARCH_LOG.md** for methodology and findings, and **results/report_*.md** for
ready-to-use report text.

## How to explore this project (new readers start here)

**1. Understand the goal (5 min).** Read this README, then skim **RESEARCH_LOG.md** — it tells the
whole story: what the code did *before*, what was changed, and *why*, with every result.

**2. See the results without running anything.** Open `results/figures/` (the charts) and
`results/report_exp1_exp2.md` / `report_exp3_exp4.md` (plain-English findings with tables).
Short version: **SIFT is the most accurate, ORB is the best speed/accuracy balance, LBP is
unusable for identity, Minutiae is good for visualization.**

**3. Try it yourself (most intuitive first).** With the dataset in place:
```bash
python apps/sift_app.py        # choose 2 = 1:N attendance, then enter e.g. 105_2.tif
python apps/minutiae_app.py    # choose 3 = draw the minutiae map (red=ending, blue=bifurcation)
python experiments/run_experiments.py --exp 3   # the full algorithm comparison table
```

**4. Read the code in this order** (each file is small and single-purpose):
| Step | File | What it contains |
|------|------|------------------|
| 1 | `core/io_utils.py` | how images and dataset paths are loaded; the fixed seed |
| 2 | `core/preprocessing.py` | the 6 preprocessing combos (C1/C2/C3 + Gabor) |
| 3 | `core/features.py` | the 4 feature extractors (SIFT, ORB, LBP, Minutiae) |
| 4 | `core/matching.py` | matching + RANSAC and the 4 scoring strategies (S1..S4) |
| 5 | `core/evaluation.py` | how EER (1:1) and Rank-1 / identification rate (1:N) are computed |
| 6 | `experiments/run_experiments.py` | ties everything together into the 4 studies |
| 7 | `apps/*.py` | self-contained interactive demos, one per algorithm |

**One-sentence pitch:** *a comparison of four classical, zero-training fingerprint algorithms
(SIFT, ORB, LBP, Minutiae) on FVC2002, evaluated for 1:1 verification (EER) and 1:N identification
(a class-attendance demo), studying preprocessing, the algorithms themselves, and scoring.*

## Dataset
FVC2002 (set B: 10 fingers × 8 impressions per database, DB1–DB4). It is licence-restricted
and therefore not committed; download it (e.g. from Kaggle) and place each database under
`fingerprints/DB1_B`, `DB2_B`, `DB3_B`, `DB4_B`.

## License
Released under the MIT License — see [LICENSE](LICENSE).
