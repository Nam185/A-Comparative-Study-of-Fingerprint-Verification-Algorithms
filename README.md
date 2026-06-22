# A Comparative Study of Fingerprint Verification Algorithms

Traditional, **zero-training** fingerprint recognition, comparing four hand-crafted
feature/matching approaches (SIFT, ORB, LBP, Minutiae) on two public datasets:
**FVC2002** (controlled accuracy benchmark) and **SOCOFing** (a large, real 1:N gallery).
The study covers preprocessing, the algorithms, scoring strategy, and a real-world
scalability test, evaluated for **1:1 verification (EER)** and **1:N identification**
(a class-attendance scenario) with **accuracy and latency** as the two primary criteria.

All experiments are reproducible (fixed seed) and every change/finding is logged in
[RESEARCH_LOG.md](RESEARCH_LOG.md). Ready-to-paste report text lives in `results/report_*.md`.

## Algorithms compared
| Method | Feature extracted | Matching |
|--------|-------------------|----------|
| **SIFT** | scale-invariant keypoints (128-D gradient descriptors) | BFMatcher + Lowe ratio + RANSAC inliers |
| **ORB** | binary keypoints (FAST + oriented BRIEF) | BFMatcher (Hamming) + ratio + RANSAC |
| **LBP** | block-based texture histogram (8×8 grid) | Chi-Square |
| **Minutiae** | ridge endings/bifurcations + orientation (Crossing Number) | dedicated geometry matcher (`minutiae_native`, Exp 5–6) or minutiae+local-descriptor hybrid (Exp 1–3) |

## The experiments
| # | Study | Dataset | Headline finding |
|---|-------|---------|------------------|
| 1 | Preprocessing (SIFT, 6 combos × 4 DBs) | FVC2002 B | DB quality dominates; C1/C2 solid, Gabor no reliable help |
| 2 | Preprocessing generalization (6 combos × 4 algos) | FVC2002 B | best combo is algorithm-dependent; doesn't transfer |
| 3 | Algorithm comparison — **EER + latency**, 1:1 & 1:N | FVC2002 B | SIFT most accurate; LBP unusable for identification |
| 4 | Scoring strategy (S1..S4) | FVC2002 B | RANSAC inlier **count** (S3) is best; ratio (S4) worst |
| 5 | SIFT vs Minutiae **at scale** (speed) | SOCOFing | accuracy saturates → Minutiae ~14× faster wins |
| 6 | SIFT vs Minutiae **accuracy** (best-vs-best) | FVC2002 B | SIFT robust to real variation; simple minutiae is not |

Plus two visualizations: a **feature-extraction showcase** and a **genuine-vs-impostor matching** figure.

## Repository layout
```
core/                shared code used by both the experiments and the demo apps
  io_utils.py          image reading, dataset paths, fixed random seed
  preprocessing.py     6 combos: C1/C2/C3 (+Gabor variants)
  features.py          SIFT / ORB / LBP / Minutiae(hybrid) extractors
  minutiae_native.py   dedicated minutiae matcher (orientation + geometry, no SIFT descriptors)
  matching.py          match + RANSAC + scoring variants S1..S4
  evaluation.py        1:1 EER, 1:N Rank-1 / identification rate
apps/                interactive demo programs (one per algorithm)
  sift_app.py  orb_app.py  lbp_app.py  minutiae_app.py
experiments/
  run_experiments.py     reproducible studies (--exp 1..6)
  visualize_extraction.py  draw SIFT/ORB/LBP/Minutiae features on a fingerprint
  visualize_matching.py    draw genuine-vs-impostor SIFT matches
results/             CSV outputs + figures/ + report_*.md (ready-to-paste report text)
  report_exp1_exp2.md  report_exp3_exp4.md  report_exp5.md  report_exp6.md
  report_feature_extraction.md  recent_changes_summary.md
fingerprints/        datasets (NOT committed — licence-restricted, see .gitignore)
RESEARCH_LOG.md      full chronological changelog + every result (Iterations 1–7)
```

## Quick start
```bash
pip install -r requirements.txt
# place the datasets (see "Datasets" below) under fingerprints/

# Reproducible experiments (each writes CSV(s) to results/ and figure(s) to results/figures/)
python experiments/run_experiments.py --exp 1   # preprocessing study (SIFT, 4 DBs)
python experiments/run_experiments.py --exp 2   # preprocessing generalization (4 algos)
python experiments/run_experiments.py --exp 3   # algorithm comparison: EER + latency, 1:1 & 1:N
python experiments/run_experiments.py --exp 4   # scoring strategy study (S1..S4)
python experiments/run_experiments.py --exp 5   # SOCOFing: SIFT vs Minutiae at scale (speed)
python experiments/run_experiments.py --exp 6   # FVC: SIFT vs Minutiae accuracy (best-vs-best)

# Visualizations (saved to results/figures/)
python experiments/visualize_extraction.py DB2_B/101_1.tif   # feature-extraction showcase
python experiments/visualize_matching.py DB2_B 101           # genuine vs impostor matches

# Interactive demos (menu: 1=evaluate EER, 2=1:N attendance, 3=outlier/visualize, 4=exit)
python apps/sift_app.py
```
Run experiments one at a time (each loads SIFT; running several at once can exhaust memory).
A fixed seed (`core.io_utils.SEED = 42`) makes every run reproducible.

## Key findings
- **Accuracy (FVC2002, real multi-impression data):** **SIFT is the most accurate** (mean EER ≈ 26 %),
  followed by ORB; **LBP is unusable for identification** (≈ 0 % identification rate — it ranks by luck
  but cannot clear a decision threshold). Sensor quality dominates: easy DB2 ≈ 6 % EER, hard/dark DB3 ≈ 33 %,
  synthetic DB4 fails for everyone.
- **Scoring:** geometric verification with the **RANSAC inlier count (S3)** beats raw count, percentage,
  and inlier-ratio.
- **Speed at scale (SOCOFing, 6 000-print gallery):** when accuracy saturates, **speed decides** — the
  dedicated Minutiae matcher is **~14× faster** per comparison than SIFT.
- **But under real capture variation (FVC), the simple Minutiae matcher is not robust** (EER ≈ 49 %,
  genuine ≈ impostor), whereas SIFT stays robust. So the real-world answer is a trade-off: SIFT for
  accuracy/robustness, Minutiae for raw speed; a production system needs an indexed minutiae matcher
  (e.g. MCC) to get both.

## How to explore this project (new readers start here)
1. **Goal (5 min):** read this README, then skim [RESEARCH_LOG.md](RESEARCH_LOG.md) for the full story.
2. **Results without running anything:** browse `results/figures/` and the `results/report_*.md` files;
   [recent_changes_summary.md](results/recent_changes_summary.md) summarizes the latest updates.
3. **Try it (most intuitive first):**
   ```bash
   python apps/sift_app.py        # choose 2 = 1:N attendance, then enter e.g. 105_2.tif
   python apps/minutiae_app.py    # choose 3 = draw the minutiae map (red=ending, blue=bifurcation)
   python experiments/visualize_matching.py DB2_B 101   # genuine (562) vs impostor (5) matches
   ```
4. **Read the code in this order** (each file is small and single-purpose):

| Step | File | What it contains |
|------|------|------------------|
| 1 | `core/io_utils.py` | image/dataset loading, the fixed seed |
| 2 | `core/preprocessing.py` | the 6 preprocessing combos (C1/C2/C3 + Gabor) |
| 3 | `core/features.py` | the SIFT/ORB/LBP/Minutiae(hybrid) extractors |
| 4 | `core/minutiae_native.py` | the dedicated minutiae matcher (orientation + geometry) |
| 5 | `core/matching.py` | matching + RANSAC and the scoring strategies (S1..S4) |
| 6 | `core/evaluation.py` | how EER (1:1) and Rank-1 / identification rate (1:N) are computed |
| 7 | `experiments/run_experiments.py` | ties everything into the 6 studies |
| 8 | `apps/*.py` | self-contained interactive demos, one per algorithm |

## Datasets
Both datasets are licence-restricted and therefore **not committed** (see `.gitignore`); download them
yourself and place them under `fingerprints/`.

- **FVC2002 set B** — 4 databases (DB1–DB4), 10 fingers × 8 real impressions each. Used for accuracy
  (EER) because it has multiple real impressions and hard imagery. Place at
  `fingerprints/DB1_B`, `DB2_B`, `DB3_B`, `DB4_B`.
- **SOCOFing** — 600 subjects × 10 fingers (6 000 real prints) + synthetically *Altered* versions
  (Easy/Medium/Hard). Used for the large-gallery 1:N speed study. Licence: noncommercial research.
  Place at `fingerprints/SOCOFing/Real` and `fingerprints/SOCOFing/Altered/Altered-{Easy,Medium,Hard}`.

## License
Released under the MIT License — see [LICENSE](LICENSE). The MIT license covers the **code only**, not
the datasets, which keep their own licences.
