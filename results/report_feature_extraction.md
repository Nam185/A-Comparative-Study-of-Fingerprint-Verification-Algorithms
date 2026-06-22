# Feature Extraction — Report Text

*Ready-to-paste English text. Figures: `results/figures/feature_extraction_showcase.png`
(all four algorithms side by side) and `feature_extraction_minutiae.png` (close-up).
Reproduce with `python experiments/visualize_extraction.py DB1_B/101_1.tif`.*

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
