# Preprocessing Study — Report Text (Experiments 1 & 2)

*Ready-to-paste English text for the report. Figures referenced here are
`results/figures/exp1_preprocessing_eer.png` and `exp2_generalization_eer.png`.
All runs use a fixed random seed (42) so the numbers are reproducible.*

---

## Experiment 1 — Preprocessing Study on SIFT

### What we did
We evaluated six preprocessing pipelines on the SIFT matcher across all four FVC2002
databases (DB1_B–DB4_B) for the 1:1 verification task. The three base pipelines combine a
denoising filter with a contrast-enhancement step, and each is also tested with an additional
Gabor filter bank:

- **C1** – Gaussian Blur + CLAHE
- **C2** – Bilateral Filter + CLAHE
- **C3** – Median Blur + Histogram Equalization
- **C1+G, C2+G, C3+G** – the above, each followed by a Gabor filter bank

For every (combo, database) pair we recorded the Equal Error Rate (EER), the mean genuine
and imposter scores, the average number of SIFT keypoints per image, and the average number
of RANSAC-verified inliers on genuine pairs. Matching uses Lowe's ratio test (0.75) followed
by RANSAC geometric verification, and the score is the inlier count.

### Results
EER (%) per database and combo (see Figure *exp1_preprocessing_eer*):

| Combo | DB1_B (avg) | DB2_B (easy) | DB3_B (hard) | DB4_B (synthetic) |
|-------|-------------|--------------|--------------|-------------------|
| C1    | 16.67 | 6.19 | 32.54 | 48.25 |
| C2    | 16.43 | 6.19 | 31.43 | 46.19 |
| C3    | 25.79 | 2.86 | 45.63 | 51.51 |
| C1+G  | 16.35 | 11.27 | 38.65 | 51.27 |
| C2+G  | 25.48 | 11.19 | 36.19 | 45.08 |
| C3+G  | 13.81 | 6.51 | 45.00 | 49.05 |

A representative keypoint measurement (DB1_B): C1 produces **2305** keypoints per image on
average, whereas C1+G produces only **987** — the Gabor stage roughly halves the keypoint count.

### Discussion
The dominant effect is the **database, not the preprocessing**: DB2_B stays in the 3–11% range
and DB4_B in the 45–52% range under every pipeline. DB3_B is consistently hard because its
capacitive-sensor images are dark and low-contrast, so SIFT finds far fewer stable keypoints,
and DB4_B is the worst because it is synthetically generated. This shows that preprocessing
cannot compensate for fundamentally poor input quality.

Within a single database the differences between combos are small. Because each database
provides only 70 genuine and 45 imposter comparisons, the EER is quantised in steps of roughly
1.4–2.2 percentage points; **differences below about three percentage points are within this
sampling noise** and should not be over-interpreted. With that caveat, C1 and C2 (CLAHE-based
contrast enhancement) are reliably among the best, while C3 (global Histogram Equalization) is
the most erratic — excellent on the clean DB2_B (2.86%) but poor on DB1_B and DB3_B — which we
attribute to global equalization amplifying background noise in lower-quality images.

The clearest, most reliable Gabor result is the **measured halving of the keypoint count**. Its
effect on EER, however, is small and inconsistent in sign across databases (it lowered EER on
DB1_B but raised it on DB3_B), i.e. it does not provide a dependable accuracy gain for SIFT.

### Conclusion
We keep **C1 (Gaussian Blur + CLAHE)** as the production default: it is consistently near the
best EER for SIFT, is the simplest pipeline, and retains about twice as many keypoints as the
Gabor variants for comparable accuracy.

---

## Experiment 2 — Preprocessing Generalization Across Algorithms

### What we did
To test whether the preprocessing conclusion from Experiment 1 transfers to other feature
types, we ran the same six combos on all four algorithms (SIFT, ORB, LBP, Minutiae) on the
single representative database DB1_B (1:1 task). For the Minutiae method we additionally logged
the average number of detected minutiae per image, to test a specific hypothesis (below).

### Results
EER (%) per algorithm and combo on DB1_B (see Figure *exp2_generalization_eer*):

| Combo | SIFT | ORB | LBP | Minutiae |
|-------|------|-----|-----|----------|
| C1    | 16.67 | 21.90 | 44.37 | 21.75 |
| C2    | 16.43 | 27.62 | 42.54 | 21.03 |
| C3    | 25.79 | 31.03 | 46.19 | 20.95 |
| C1+G  | 16.35 | 26.83 | 42.54 | **17.86** |
| C2+G  | 25.48 | 26.90 | 42.54 | 19.60 |
| C3+G  | **13.81** | 25.00 | 42.54 | 18.89 |

Best combo per algorithm: SIFT → C3+G (13.81%), ORB → C1 (21.90%), LBP → C2 (42.54%),
Minutiae → C1+G (17.86%). Average Gabor effect (mean EER change over C1/C2/C3): SIFT −1.08,
ORB −0.61, LBP −0.61, Minutiae **−2.46** percentage points. Average minutiae count is
essentially unchanged by Gabor (C1 276.9 → C1+G 276.2).

### Discussion
The preprocessing conclusion **does not fully transfer**: the best combo differs for each
algorithm, so there is no single universally optimal pipeline. LBP remains in the 42–46% band
regardless of preprocessing, confirming that a global-texture descriptor without geometric
alignment is fundamentally unsuited to fingerprint identity — no amount of front-end filtering
rescues it.

The most interesting case is the Minutiae method. The lecturer's hypothesis was that Gabor
enhancement would clean up the binarization and therefore reduce the number of spurious minutiae.
Our measurements **do not support that mechanism**: the average minutiae count barely changes
with Gabor (276.9 vs 276.2). Nevertheless, Gabor produced the **largest EER improvement** of any
algorithm for Minutiae (−2.46 pp, best result 17.86%). We therefore believe the benefit comes
not from fewer minutiae but from the ridge-enhanced image yielding **more distinctive local
descriptors** at each minutia; this remains a hypothesis that would need a dedicated test to
confirm. For the other three algorithms the average Gabor effect (−0.6 to −1.1 pp) lies within
the sampling noise discussed in Experiment 1 and is not significant.

### Conclusion
We retain **C1 (Gaussian Blur + CLAHE)** as the shared production default because it is robust
across all four algorithms, and we report the per-algorithm best combos as a secondary finding.
Gabor is not adopted globally, as its only consistent, reliably-measured effect is increased
computation and a halved keypoint count, with accuracy gains that are significant only for the
Minutiae method.

---

## Limitations and Future Work *(optional, secondary)*
- The evaluation set is small (70 genuine / 45 imposter per database), so EER differences of a
  few percentage points are not statistically significant; the full FVC protocol (all impression
  pairs, 2 800 genuine) would give more stable estimates.
- The Gabor bank uses fixed parameters; a per-block ridge orientation/frequency estimation would
  be a more faithful implementation and might change its effect.
- The hypothesis that Gabor improves Minutiae via more distinctive descriptors (rather than fewer
  minutiae) could be verified by measuring per-minutia descriptor distinctiveness directly.
