# Feature Extraction & Matching — Report Text

*Ready-to-paste English text. Figures (in `results/figures/`):
`feature_extraction_showcase_DB1_B_101_1.png` (all four algorithms side by side),
`feature_extraction_showcase_DB2_B_101_1.png` (clean input),
`feature_extraction_showcase_DB3_B_101_1.png` (hard/dark input),
`feature_matching_genuine_vs_impostor.png` (genuine vs impostor matches).
Reproduce: `python experiments/visualize_extraction.py DB1_B/101_1.tif` and
`python experiments/visualize_matching.py DB2_B 101`.*

---

## Feature Extraction (the core of the traditional pipeline)

For hand-crafted (non-deep-learning) fingerprint recognition, **feature extraction is the decisive
stage** — everything downstream depends on whether the extractor finds repeatable, discriminative
features on the ridge structure. To verify that each algorithm extracts features **correctly and
well**, we run the actual extractors on a sample fingerprint and visualise their output
(Figure *feature_extraction_showcase*). Every method shares the same front-end preprocessing
(Normalize → Gaussian Blur → CLAHE), which raises ridge/valley contrast before extraction.

**SIFT — scale-invariant keypoints.** SIFT detects blob/corner structures across multiple scales and
assigns each keypoint a location, scale and orientation (shown as the rich circles). On this sample
it produces **~1 416 keypoints**, densely covering the ridge region and avoiding the background,
which confirms the detector is responding to genuine ridge structure rather than noise. Each keypoint
carries a 128-D gradient descriptor used for matching.

**ORB — fast binary keypoints.** ORB combines a FAST corner detector with an oriented BRIEF binary
descriptor. It yields **~1 500 keypoints** (capped by `nfeatures=1500`), again concentrated on the
ridges. Its binary descriptors are cheaper to compute and match than SIFT's, at the cost of less
distinctiveness on smooth ridge patterns.

**LBP — local texture encoding.** LBP replaces each pixel by a code describing how its neighbourhood
compares to it (uniform pattern, radius 2, 16 sampling points). The texture map (panel 5) shows the
ridge flow encoded as discrete patterns; the final feature is the concatenation of per-cell
histograms over an 8×8 grid. This captures *texture* but not the *geometry* of specific ridge points.

**Minutiae — ridge endings and bifurcations.** The minutiae extractor binarises the enhanced image,
thins the ridges to a 1-pixel skeleton, and applies the **Crossing Number** rule: a skeleton pixel
with one neighbour is a **ridge ending** (red) and one with three neighbours is a **bifurcation**
(blue); a local ridge **orientation** (green tick) is attached to each. On this sample it extracts
**158 minutiae**, located on the ridge endings/forks as expected. (Endings dominate bifurcations here
because broken ridges in the binarised image create extra endings — a known sensitivity of the
Crossing Number method, mitigated by the ROI mask and spurious-minutiae removal.)

**Verification.** In every panel the extracted features fall on the segmented fingerprint region with
substantial, sensible counts (≈1 400–1 500 keypoints; ≈150 minutiae), confirming that all four
extractors run correctly and respond to real ridge structure. This visual check is the foundation
for trusting the downstream matching results in Experiments 1–6.

## Effect of sensor quality on extraction (clean vs hard input)
Running the same extractors on a **clean** database (DB2_B, optical 500 dpi) and a **hard** one
(DB3_B, a dark, low-contrast capacitive sensor) makes the central finding of this study —
*sensor/image quality dominates accuracy* — visible already at the extraction stage:

- **DB2_B (clean):** the preprocessed ridges are crisp, the LBP texture map follows the ridge flow
  cleanly, and the features sit tightly on the ridges (≈2 435 SIFT keypoints, 321 minutiae). This is
  what "extraction working well" looks like.
- **DB3_B (hard, dark):** CLAHE has to amplify a weak signal, which also amplifies background noise.
  The LBP map becomes speckled, SIFT/ORB keypoints spill into the noisy background, and the minutiae
  set gains many **spurious bifurcations** clustered in noisy regions (92 vs 25 on DB2). The feature
  *counts* stay comparable, but their *quality and repeatability* drop — which is exactly why DB3
  matching is far worse (1:1 EER ≈ 32 % vs ≈ 6 % on DB2). In other words, the accuracy gap between
  sensors originates in the extraction stage, not in the matcher.

## Matching: separating genuine from impostor
Extraction alone is not enough — the system must *discriminate*. Figure
*feature_matching_genuine_vs_impostor* draws the geometrically consistent (RANSAC-verified) SIFT
matches for two pairs from DB2_B:

- **Genuine pair** (two impressions of the same finger): **562 consistent matches**, and the lines
  are roughly parallel — a coherent geometric transform between the two prints.
- **Impostor pair** (two different fingers): only **5 consistent matches**, scattered and incoherent.

This ~100× gap is the visual proof that the pipeline does not merely "find points" but actually
**separates same-finger from different-finger pairs**, which is the basis of every EER and Rank-1
number reported in this study.
