# Real-World Scalability — Report Text (Experiment 5, SOCOFing)

*Ready-to-paste English text. Figures: `results/figures/exp5_latency_scaling.png`,
`exp5_accuracy_vs_difficulty.png`. Tables: `exp5_preprocessing_tuning.csv`,
`exp5_socofing_latency.csv`, `exp5_socofing_accuracy.csv`. Fixed seed (42).*

---

## Experiment 5 — SIFT vs Minutiae at scale (SOCOFing)

### What we did
The FVC2002 study used small galleries (10 fingers). To test our conclusions on a
**large, real database** where 1:N speed becomes the binding constraint, we use **SOCOFing**
(600 subjects × 10 fingers = 6 000 real fingerprints, plus synthetically *altered* versions in
three difficulty levels: Easy, Medium, Hard). We compare the two strongest approaches, **SIFT**
and a **dedicated Minutiae matcher** that uses minutiae geometry and orientation only — *not*
SIFT descriptors. The protocol is closed-set identification: enroll the *Real* images, query with
their *Altered* versions; a query is correct if its true finger is the top match in the gallery.

Three differences from the controlled study, all deliberate:
1. **Each algorithm uses its own best preprocessing** (tuned separately), not a shared pipeline.
2. SOCOFing images are tiny (~96×103 px), so preprocessing includes **upscaling (×2–×3)**.
3. The headline criterion is **speed at scale**, because real galleries are large.

### (B) Per-method preprocessing tuning
We swept four pipelines per method (`x2+C1`, `x3+C1`, `x3+C1G`, `x3+C2`) and measured EER on an
Altered-Hard validation subset. **Every candidate produced perfect genuine/impostor separation
(EER = 0 %)** — on SOCOFing the genuine scores dwarf the impostor scores (e.g. SIFT 636 vs 10;
Minutiae 47 vs 4), so the preprocessing choice does not affect accuracy. Because accuracy is
saturated, the tuning selects the **cheapest option, `x2+C1` (×2 upscale + CLAHE)**, for both
methods — which also minimises latency, consistent with the goal of this experiment.

### (A) Latency and scaling
| Algorithm | Features/img | Extraction (ms/img) | **Match (ms/comparison)** |
|-----------|--------------|---------------------|----------------------------|
| SIFT      | ~1292 keypoints | 25.3 | **15.1** |
| Minutiae  | ~141 minutiae   | 53.0 | **0.59** |

Minutiae matches **~25× faster** than SIFT, because it compares ~141 points instead of ~1292
keypoints. Since a 1:N query costs `extract + N × match`, this gap dominates as the gallery grows
(Figure *exp5_latency_scaling*, log–log):

| Gallery N | SIFT (s/query) | Minutiae (s/query) |
|-----------|----------------|---------------------|
| 100       | 1.7   | 0.11 |
| 1 000     | 16.9  | 0.60 |
| 6 000     | 101   | 3.4  |
| 60 000    | ~1014 (≈17 min) | 33 |

### (C) Identification accuracy
With a 100-identity gallery, **both methods achieve 100 % Rank-1 and 100 % identification rate
at every alteration level — Easy, Medium and Hard** (Figure *exp5_accuracy_vs_difficulty*; the two
lines overlap at 100 %). The SOCOFing alterations (obliteration, central rotation, z-cut) preserve
enough ridge structure that matching a finger to its own altered version is unambiguous against
distinct other fingers. **Accuracy does not separate the two methods.**

### Discussion
The result is decisive precisely *because* accuracy saturates. When two methods are equally
accurate, the deployment choice is governed by efficiency, and here Minutiae is roughly **25×
faster per comparison**. For a real attendance/identification system with a large enrolment
database, SIFT becomes impractical (≈17 minutes per query at N = 60 000), whereas the Minutiae
matcher stays usable (≈33 s, and far less with indexing). We also note that the dataset's tiny
images did **not** prevent high accuracy after upscaling — classical matching is more robust to
low resolution here than expected, likely because the alterations keep the global ridge flow intact.

### Conclusion
For the real-world, large-gallery 1:N problem, the **dedicated Minutiae matcher is the better
choice**: it matches SIFT's accuracy on SOCOFing (both ~100 %) while being about **25× faster**,
which is the property that actually determines feasibility at scale. SIFT remains the safer pick
only when the gallery is small and maximum robustness on hard, low-quality single comparisons is
required.

### Limitations and future work *(optional)*
- At a 100-identity gallery accuracy is saturated (100 %), so it cannot rank the two methods on
  accuracy; a much larger gallery would be needed to surface errors, but evaluating SIFT there is
  itself impractical (hours of compute) — which reinforces the scalability argument.
- SOCOFing's "second impressions" are synthetic alterations, not natural re-captures; results may
  differ on a dataset with multiple real impressions per finger.
- A minutiae **index** (e.g. hashing local minutiae structures) would replace the linear
  `N × match` scan with sub-linear search, widening the speed advantage further.
