# Accuracy Under Real Variation — Report Text (Experiment 6, FVC2002 set B)

*Ready-to-paste English text. Figure: `results/figures/exp6_accuracy_fvcB.png`.
Table: `results/exp6_accuracy_fvcB.csv`. Fixed seed (42). This experiment supplies the
ACCURACY half of the real-world 1:N picture, complementing the SPEED result of Experiment 5.*

---

## Experiment 6 — SIFT vs the dedicated Minutiae matcher on FVC2002 set B

### What we did
Experiment 5 showed that on the large SOCOFing gallery both methods reached 100 % accuracy, so
only speed could be compared there. To compare **accuracy** properly we use **FVC2002 set B**,
which — although it has only 10 fingers per database — has two decisive advantages for measuring
accuracy: it contains **multiple genuine impressions per finger** (8 real captures, with natural
rotation, pressure and partial-overlap variation) and **hard imagery** (the EER is far from
saturated). We run 1:1 verification (genuine = impressions of the same finger, impostor = different
fingers) and report the **Equal Error Rate (EER)**. This is a **best-vs-best** comparison: SIFT uses
its best preprocessing (C1), and the **dedicated Minutiae matcher** — minutiae described by their
orientation (θ) and local geometry, matched with RANSAC, **without any SIFT descriptors** — uses its
own best preprocessing (selected as C1+G by a small sweep).

### Results — 1:1 EER (%) per database
See Figure *exp6_accuracy_fvcB*.

| Database | SIFT | Minutiae-native |
|----------|------|------------------|
| DB2_B (easy)      | **6.19**  | 44.13 |
| DB1_B (average)   | **16.67** | 40.56 |
| DB3_B (hard)      | **32.54** | 52.78 |
| DB4_B (synthetic) | **48.25** | 59.60 |
| **Mean**          | **25.91** | **49.27** |

The genuine/impostor score gap tells the story: for SIFT the genuine average is ~143 vs ~5 for
impostors (clean separation), whereas for the Minutiae matcher the genuine average (~4.5) is barely
above the impostor average (~4.0) — almost **no separation**, i.e. close to random.

### Discussion
On FVC, **SIFT is decisively more accurate** (mean EER 25.9 % vs 49.3 %). The dedicated Minutiae
matcher, which worked perfectly on SOCOFing, almost fails to separate genuine from impostor pairs
here. The reason is the **type of genuine pair**:

- On **SOCOFing** a genuine pair is *Real vs Altered* — the altered image is a synthetic
  transformation of the very same capture, so the two minutiae sets are nearly identical and align
  trivially. The 100 % accuracy in Experiment 5 reflects this near-duplicate condition.
- On **FVC** a genuine pair is *two independent real captures* of the finger, with real rotation,
  distortion and only partial overlap. Many minutiae appear in one impression but not the other, and
  spurious minutiae differ between captures. A simple minutiae matcher loses the correspondence, so
  genuine pairs accumulate roughly as few consistent matches as impostor pairs.

SIFT survives this because its rich gradient descriptors plus a homography (RANSAC) model the real
geometric distortion, while a few-nearest-neighbour minutiae descriptor is too fragile when minutiae
are missing or added.

**Robustness check (to rule out configuration / segmentation artefacts).** Because the matcher's
geometric parameters were first set on the upscaled SOCOFing images (~192 px) and FVC images are
larger (374–560 px), we verified the result is not merely a mis-configuration:
1. *Extraction is healthy on FVC* — it yields ~100–200 minutiae per image, the expected range.
2. *Segmentation was improved* — the ROI mask was upgraded (variance **plus** gradient-orientation
   coherence, then largest connected component) so that minutiae are no longer detected in textured
   background; despite removing those spurious points, EER did **not** improve (it stayed ~40–53 %),
   ruling out background noise as the cause.
3. *Loosening the geometry* (Lowe ratio up to 0.95, RANSAC threshold up to 20 px) and *a
   scale-invariant descriptor* (neighbour distances normalised by their mean) both kept EER in the
   ~35–46 % range, still with genuine ≈ impostor.

Across all of these the genuine and impostor score distributions stayed overlapped, so the failure is
a **structural limitation of this simple matcher** — it loses minutiae correspondence when the two
captures differ — **not** a mis-scaled parameter or a segmentation leak. It is, however, **not** a
limitation of minutiae methodology in general (see Limitations).

This result also **re-frames Experiment 5 honestly**: the Minutiae matcher's apparent accuracy on
SOCOFing was a property of the near-duplicate test pairs, not of robustness to real variation.

### Conclusion — the complete real-world 1:N picture
Combining both experiments gives the full trade-off:

| Criterion | Best method | Evidence |
|-----------|-------------|----------|
| **Accuracy under real variation** | **SIFT** | FVC mean EER 25.9 % vs 49.3 % (Exp 6) |
| **Speed at scale** | **Minutiae** | ~14× faster match; usable at N = 60 000 (Exp 5) |

So neither method is ideal for real-world 1:N on its own: **SIFT is accurate and robust to real
capture variation but too slow for a very large gallery**, while **a simple Minutiae matcher is fast
but not robust to real inter-impression distortion**. The practical implication is that a deployable
large-scale system needs a *production-grade* minutiae approach (a robust descriptor such as MCC plus
an index for sub-linear search) to obtain SIFT-level accuracy at minutiae-level speed.

### Limitations and future work *(optional)*
- Our Minutiae matcher is a deliberately simple, from-scratch implementation. Established matchers
  (NBIS Bozorth3, MCC) reach single-digit EER on FVC; our ~44 % reflects the simplified matcher, not
  the ceiling of minutiae methodology. Reproducing a stronger matcher is the natural next step.
- FVC set B has few fingers, so 1:1 EER (not 1:N Rank-1) is the meaningful accuracy metric here; the
  large-gallery accuracy question can only be answered with a faster, indexed matcher.
