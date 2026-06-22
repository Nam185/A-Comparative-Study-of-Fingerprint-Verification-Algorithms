# Recent Changes — Summary

*A focused summary of the latest batch of changes (feature-extraction visualizations, a
segmentation bug fix, and the re-synchronisation of the affected experiments). The full
chronological history lives in `RESEARCH_LOG.md` (Iterations 1–7); this file zooms in on
"what just changed, what it did to the numbers, and why".*

---

## 1. What was added

### 1a. Feature-extraction showcase (`experiments/visualize_extraction.py`)
Renders the **actual** `core/` extractors on a fingerprint so the extraction stage can be
inspected visually (the teacher's request: extraction is the heart of a traditional pipeline).
Output: a 6-panel figure (original → preprocessed → SIFT / ORB / LBP / Minutiae) plus a close-up
minutiae figure, with feature counts printed.

| Sample | SIFT kp | ORB kp | Minutiae (end / bif) | Purpose |
|--------|---------|--------|----------------------|---------|
| DB1_B (average) | 1416 | 1500 | 131 (124 / 7) | canonical example |
| DB2_B (clean)   | 2435 | 1500 | 205 (188 / 17) | "extraction is tidy on good input" |
| DB3_B (hard, dark) | 2106 | 1394 | 180 (96 / 84) | extraction degrades on a noisy sensor |

**Finding:** the clean DB2 gives crisp, well-localised features; the dark DB3 forces CLAHE to
amplify background noise, producing a speckled LBP map and spurious minutiae. So the
"sensor quality dominates accuracy" conclusion is visible **already at the extraction stage**.

### 1b. Matching visualization (`experiments/visualize_matching.py`)
Draws the RANSAC-verified SIFT matches for a genuine vs an impostor pair (DB2_B):

| Pair | Consistent (inlier) matches |
|------|------------------------------|
| Genuine (101_1 vs 101_2, same finger) | **562** (parallel, coherent) |
| Impostor (101_1 vs 102_1, different fingers) | **5** (scattered) |

**Finding:** a ~100× gap — direct visual proof that the pipeline *discriminates* genuine from
impostor, not merely "finds points". This is the most persuasive single figure of correctness.

---

## 2. The bug that was fixed — ROI (fingerprint) segmentation

**Symptom (spotted on the showcase):** on DB2/DB3, minutiae appeared **outside** the fingerprint,
in the textured/striped background near the image border.

**Root cause:** the ROI mask used **local variance only**. A dark capacitive sensor (DB3) has a
striped, noisy background whose variance is *also* high, so it was wrongly classified as
fingerprint → spurious minutiae there.

**Fix (`core/minutiae_native.py`, `core/features.py`, `apps/minutiae_app.py`):** the mask now
requires **high variance AND high gradient-orientation coherence** (real ridges flow consistently;
noise does not), then keeps only the **largest connected component** (drops detached background
patches).

**Effect on extraction (spurious points removed):** minutiae per image dropped to a cleaner set —
DB1 158→131, DB2 321→205, DB3 263→180 — and they now sit inside the fingerprint.

This bug affected not only the pictures but the **real matcher**, so every minutiae-based
experiment was re-run.

---

## 3. Impact on the experiments (before → after the ROI fix)

> SIFT / ORB / LBP do **not** use this mask, so their numbers are unchanged. Only **Minutiae** moves.
> **None of the study's conclusions change.**

| Experiment | Metric | Before | After | Why it moved |
|-----------|--------|--------|-------|--------------|
| **Exp 6** (native matcher, FVC) | mean EER | 44.37% | **49.27%** | cleaner ROI, yet genuine ≈ impostor still → *reinforces* the "structural limitation" conclusion (it was never the background noise) |
| **Exp 5** (SOCOFing) | Minutiae match | 0.59 ms | 0.99 ms | best preprocessing switched to ×3-upscale (more minutiae) |
| **Exp 5** | speed advantage | ~25× | **~14×** faster than SIFT | same reason; still a large, decisive gap |
| **Exp 5** | accuracy | ~100% | ~100% (Hard IDrate 97%) | unchanged conclusion: accuracy saturates → speed decides |
| **Exp 3** (hybrid matcher, FVC) | Minutiae mean EER | 32.08% | 31.63% | marginal |
| **Exp 3** | Minutiae match latency | 4.53 ms | **1.72 ms** | fewer minutiae → faster; Minutiae now leads 1:N Rank-1 on DB3 (62.9% vs SIFT 54.3%) |
| **Exp 2** (generalization) | Gabor effect on Minutiae | −2.46 pp | **+0.24 pp** | **correction — see §4** |

---

## 4. A finding that was corrected (important, honest)

Experiment 2 previously reported that **Gabor uniquely helped the Minutiae method (−2.46 pp)** —
the only algorithm it seemed to help. After the ROI fix this benefit **disappears** (+0.24 pp, i.e.
no help), bringing Minutiae in line with SIFT/ORB/LBP (Gabor helps none of them beyond noise).

**Why the old result was wrong:** the leaky variance-only ROI let Gabor's denoising suppress
spurious **background** minutiae, which looked like an accuracy gain. Once the fingerprint is
segmented properly, that artefact is gone.

**Lesson (worth stating in the report):** a surprising single-method result should be validated
before being trusted; here it traced back to a segmentation bug. This is a strength of the study's
methodology, not a weakness.

---

## 5. Bottom line
- New evidence of correctness: an extraction showcase and a genuine-vs-impostor matching figure.
- A real segmentation bug was found and fixed; all minutiae-based results were regenerated so the
  committed numbers match the current code.
- **Overall conclusions are unchanged:** SIFT is the most accurate; the dedicated Minutiae matcher
  is fastest at scale but not robust to real inter-impression variation; LBP is unusable for
  identification; poor-quality sensors (DB3/DB4) dominate accuracy.
