# Algorithm Comparison & Scoring Study — Report Text (Experiments 3 & 4)

*Ready-to-paste English text. Figures: `results/figures/exp3_eer_algo_x_db.png`,
`exp3_roc_db1.png`. Tables come from `exp3_algo_x_db_1to1.csv`,
`exp3_algo_x_db_1toN.csv`, `exp4_scoring.csv`. Fixed seed (42), production combo C1.*

---

## Experiment 3 — Full Algorithm Comparison (4 algorithms × 4 databases, both tasks)

### What we did
Using the production preprocessing pipeline (C1: Gaussian Blur + CLAHE), we benchmarked all
four algorithms (SIFT, ORB, LBP, Minutiae) on all four FVC2002 databases for **both** tasks:

- **1:1 verification** — reported as Equal Error Rate (EER).
- **1:N identification** (the "class attendance" use case) — reported as **Rank-1 accuracy**
  (does the highest-scoring template belong to the correct finger?) and **identification rate**
  (correct AND top score above the algorithm's fixed acceptance threshold). EER is not used for
  1:N. Each database is enrolled with impression `_1` of its 10 fingers, and queried with the
  remaining impressions `_2.._8` (70 queries).

### Results — 1:1 verification (EER %)
See Figure *exp3_eer_algo_x_db*.

| Database | SIFT | ORB | LBP | Minutiae |
|----------|------|-----|-----|----------|
| DB2_B (easy)        | **6.19**  | 12.94 | 46.19 | 26.19 |
| DB1_B (average)     | **16.67** | 21.90 | 44.37 | 21.75 |
| DB3_B (hard)        | **32.54** | 32.78 | 37.46 | 32.78 |
| DB4_B (synthetic)   | 48.25 | **40.16** | 46.90 | 47.62 |

### Results — 1:N identification (Rank-1 % / identification rate %)

| Database | SIFT | ORB | LBP | Minutiae |
|----------|------|-----|-----|----------|
| DB2_B (easy)      | **91.4 / 88.6** | 85.7 / 60.0 | 45.7 / 0.0 | 78.6 / 74.3 |
| DB1_B (average)   | **80.0 / 72.9** | 70.0 / 51.4 | 24.3 / 1.4 | 72.9 / 42.9 |
| DB3_B (hard)      | 54.3 / 32.9 | 44.3 / 21.4 | 50.0 / 0.0 | **57.1 / 42.9** |
| DB4_B (synthetic) | 17.1 / 1.4 | 15.7 / 0.0 | **21.4** / 0.0 | 20.0 / 14.3 |

### Discussion
**SIFT is the best overall method**, winning the 1:1 EER on the three real-sensor databases and
the 1:N task on the easy and average databases (Rank-1 91.4% on DB2_B). The ROC curve on DB1_B
(Figure *exp3_roc_db1*) shows the same ordering visually: SIFT sits highest in the top-left
corner, Minutiae and ORB are close behind, and **LBP hugs the diagonal "chance" line**, confirming
it is barely better than random guessing for fingerprint identity.

The most informative result is the **gap between Rank-1 accuracy and identification rate** in 1:N.
Rank-1 only asks whether the correct finger is the top match; the identification rate additionally
requires the score to clear the security threshold. For SIFT on DB1_B this gap is 80.0% → 72.9%:
a few correct matches are rejected because their score is below threshold. This is the practical
face of the FAR/FRR trade-off — a stricter threshold rejects impostors but also some genuine users.

Two findings stand out:
- **LBP is unusable in practice.** Although its Rank-1 reaches 24–50%, its identification rate is
  essentially **0%** on every database: LBP similarity scores almost never reach a threshold that
  would also reject impostors. It can sometimes rank the right person first by luck, but it cannot
  make a confident accept/reject decision.
- **The synthetic database DB4_B defeats every method** (Rank-1 15–21%, only just above the 10%
  chance level for 10 candidates). Classical hand-crafted features rely on real ridge structure
  that the synthetic generator does not reproduce faithfully.

We also note that on the hard DB3_B the four methods converge (1:1 EER 32–37%), and **Minutiae
slightly leads the 1:N Rank-1 (57.1%)** there; on such low-contrast capacitive images the explicit
ridge-ending detection is competitive with SIFT. (These small leads are within the dataset's
sampling noise; see the note in Experiment 1.)

### Conclusion
For a real attendance system we recommend **SIFT** (highest Rank-1 and identification rate on
realistic data), with **Minutiae** as a defensible second choice that is competitive on hard
images. **LBP should not be used for identification.** No method works on synthetic data.

---

## Experiment 4 — Scoring Strategy Study (SIFT and ORB)

### What we did
The matching score can be computed in several ways. We compared four scoring variants for SIFT
and ORB on DB1_B (average) and DB3_B (hard), keeping everything else fixed (combo C1, ratio test,
seed 42). The RANSAC geometry is computed once per image pair and all four scores are derived from
it, so the comparison is exactly controlled.

- **S1** – absolute good-match count (after Lowe's ratio test).
- **S2** – percentage of matches: good matches / min(keypoints) × 100.
- **S3** – RANSAC inlier count (our current production score).
- **S4** – RANSAC inlier *ratio*: inliers / good matches × 100.

### Results (EER %)

| Algo | Database | S1 | S2 | S3 (current) | S4 |
|------|----------|----|----|--------------|----|
| SIFT | DB1_B | 20.00 | 22.54 | **16.67** | 24.37 |
| SIFT | DB3_B | 38.57 | 40.00 | **32.54** | 39.29 |
| ORB  | DB1_B | 25.08 | 25.08 | **21.90** | 33.10 |
| ORB  | DB3_B | 38.17 | 38.17 | **32.78** | 35.63 |

### Discussion
**S3 (RANSAC inlier count) is the best scoring strategy in all four cases**, which justifies the
evolution of our design: S1 (raw match count) is degraded by accidental matches between different
fingers; S2 (percentage) normalises by keypoint count but still includes geometrically
inconsistent matches; S3 adds geometric verification and counts only matches that fit one
consistent transform, giving the cleanest genuine/impostor separation.

Importantly, **S4 (inlier ratio) does NOT beat S3 — it is the worst or near-worst variant.**
This is a useful negative result. Dividing inliers by the number of good matches discards the
*magnitude* of evidence: a genuine pair has many inliers, but an impostor pair that happens to
have only a handful of matches which are mostly geometrically consistent receives an inflated
ratio. This is visible in the scores — for SIFT on DB1_B the impostor average jumps from 5.47
(S3) to 32.70 (S4) — collapsing the genuine/impostor gap and raising the EER.

### Conclusion
We retain **S3 (RANSAC inlier count)** as the production score. Geometric verification is
essential, but the raw inlier *count* (evidence magnitude) must be preserved; normalising it away
(S4) hurts accuracy.

---

## Limitations and Future Work *(optional, secondary)*
- The 1:N fixed thresholds are taken from the 1:1 study and are not separately tuned per database;
  a per-database threshold would raise the identification rates, especially on DB3_B.
- All numbers use 70/45 comparisons per database, so small leads (e.g. Minutiae vs SIFT on DB3_B)
  are not statistically significant.
- A score-fusion experiment (e.g. SIFT + Minutiae) could combine SIFT's accuracy on easy data with
  Minutiae's robustness on hard capacitive images.
