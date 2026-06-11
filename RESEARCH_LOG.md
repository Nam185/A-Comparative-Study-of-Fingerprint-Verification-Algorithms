# Research Log — A Comparative Study of Fingerprint Verification Algorithms

This document is the **baseline framework** for the project. It records every change
made to the pipeline (Preprocessing → Feature Extraction → Matching → Evaluation),
the **before vs. after** state, the measured results, and **why** each change was made.
Use the "Iteration template" at the bottom to log future experiments the same way.

Dataset: **FVC2002** (`fingerprints/DB1_B..DB4_B`), each DB = 10 fingers (ID 101–110) × 8 impressions.

---

## 0. Experimental protocol (kept constant so results are comparable)

- **1:1 Verification**
  - Genuine pairs: impression `_1` vs `_2.._8` of the **same** finger → 70 pairs / DB.
  - Imposter pairs: impression `_1` of a finger vs `_1` of **every other** finger → 45 pairs / DB.
  - Metric: **FAR, FRR, EER** (Equal Error Rate). Lower EER = better.
- **1:N Identification ("Class Attendance")**
  - Enroll impression `_1` of 10 fingers as templates.
  - A query fingerprint is matched against all templates; highest score above a
    threshold → that ID, otherwise → "Imposter".
- Default evaluation DB: **DB1_B**.

---

## 1. PREPROCESSING — combo comparison

### Before (original code)
- Only **one** combo was hard-coded in all three modules:
  - **Combo 1 = Normalize (min-max) → Gaussian Blur (3×3) → CLAHE (clip 2.0, tiles 8×8)**.
- `result.docx` listed Combo 2 and Combo 3 as *planned* but they were **not implemented/measured**.

### After (systematic experiment added)
Four combos were implemented and benchmarked with the SIFT+RANSAC matcher:

| Combo | Definition | DB1_B EER | DB3_B EER | Verdict |
|-------|------------|-----------|-----------|---------|
| **C1** | Gaussian Blur + CLAHE (original) | 16.67% | 32.54% | Good, simple — current default |
| **C2** | Bilateral Filter + CLAHE | **16.43%** | **31.43%** | **Best** (marginal) — edge-preserving denoise |
| **C3** | Median Blur + Histogram Equalization | 25.79% | 45.63% | **Worst** — global HistEq amplifies noise |
| **C4** | Combo 1 + Gabor filter bank | 16.35% | 38.65% | **No improvement** (hurts DB3_B) |

**Conclusions for the report**
- **C2 (Bilateral + CLAHE)** is marginally the best; **C1** is nearly identical and simpler → both defensible.
- **C3** is the weakest: plain `equalizeHist` over-amplifies background noise.
- **Gabor (C4) did NOT help.** A fixed-parameter Gabor bank lowers the number of stable SIFT
  keypoints (DB1_B: 143 → 51 avg matches). Real Gabor ridge enhancement needs per-block
  **ridge orientation + frequency estimation**, which is outside this project's scope.
  → This is a valid *negative result*, not a failure.

> Note: the production modules still use **Combo 1** as the preprocessing step. The combo
> table above is a benchmark to justify that choice.

---

## 2. FEATURE EXTRACTION + MATCHING — per algorithm

### 2.1 SIFT (`apps/sift_app.py`)

| | Before | After |
|---|--------|-------|
| Feature | SIFT descriptors | SIFT descriptors (unchanged) |
| Matching | BFMatcher + Lowe ratio 0.8 | BFMatcher + Lowe ratio 0.75 **+ RANSAC homography** |
| Score | `good_matches / min_keypoints × 100` (%) | **number of RANSAC inliers** (geometric matches) |
| `process_single_image` returns | `descriptors` | `(keypoints, descriptors)` tuple (needed for geometry) |
| **EER (DB1_B)** | **26.90%** | **16.67%** ✅ |

**Why:** The original score counted descriptor matches but ignored their *geometry*. Two
different fingers still share a few accidental matches. **RANSAC** keeps only matches that
follow one consistent transform → imposter scores collapse (avg 2.5 → max ~9) while genuine
scores stay high (avg → 143). This is the single biggest accuracy gain.

### 2.2 ORB (`apps/orb_app.py`)

| | Before | After |
|---|--------|-------|
| Feature | ORB, `nfeatures=1000` | ORB, `nfeatures=1500` |
| Matching | BFMatcher(HAMMING) + ratio 0.7 | BFMatcher(HAMMING) + ratio 0.8 **+ RANSAC** |
| Score | `good_matches / min_keypoints × 100` (%) | **RANSAC inlier count** |
| **EER (DB1_B)** | **29.29%** | **21.90%** ✅ |

**Why:** Same RANSAC verification as SIFT. ORB improves less because its binary descriptors
and corner detector are weaker on smooth ridge patterns → ORB stays the weakest *keypoint* method.

### 2.3 LBP (`apps/lbp_app.py`)

| | Before | After |
|---|--------|-------|
| Feature | **Global** LBP histogram (whole image, 18 bins) | **Block-based** LBP: 8×8 grid, per-cell histograms **concatenated** |
| Matching | `compareHist` Correlation × 100 | **Chi-Square** distance → similarity 0..100 |
| Image size | original | resized to 256×256 (so cells align) |
| **EER (DB1_B)** | **42.78%** | **~44%** (≈ unchanged) |

**Why:** A single global histogram cannot encode identity (all fingerprints share similar
ridge texture → genuine 99.2 vs imposter 98.8, almost random). Block-based LBP is the
textbook-correct upgrade, but EER stays ~44% because LBP has **no geometric alignment** —
misaligned prints compare mismatched cells. **Honest finding: LBP is the wrong tool for
fingerprint *identity*; it is kept as a baseline to contrast against SIFT/ORB.**

### 2.4 Minutiae & Bifurcation (`apps/minutiae_app.py`) — NEW (teacher's suggestion)

| | Status |
|---|--------|
| Method | Binarize → thinning (skeleton) → **Crossing Number** (CN=1 ending, CN=3 bifurcation) → spurious removal (ROI mask + de-duplication) |
| Matching | minutiae as keypoints → local SIFT descriptor → ratio test + RANSAC |
| **EER (DB1_B)** | **~21%** (DB3_B ~31.6%) |
| Demo value | Visualization: red = ridge ending, blue = bifurcation (menu option 3) |

**Why / finding:** Implemented because the lecturer suggested it. **It does NOT beat SIFT
(21% vs 16.7%)** because minutiae extraction is highly sensitive to binarization quality —
broken ridges create many spurious endings. Its real value is (a) the explainable classical
algorithm and (b) the minutiae map for the demo.

---

## 3. 1:1 VERIFICATION — what changed

| Aspect | Before | After |
|---|--------|-------|
| Score unit | percentage (0–100%) | inlier/minutiae **count** (SIFT/ORB/Minutiae); 0–100 similarity (LBP) |
| Threshold scan | `range(0, 100)` | `range(0, 100)` (counts) / `linspace(min,max)` for LBP |
| Report fields | EER, latency, optimal threshold | **+ genuine avg & imposter avg** (shows separation) |
| Result (DB1_B) | SIFT 26.9 / ORB 29.3 / LBP 42.8 | SIFT **16.7** / ORB **27.0** / LBP ~44 / Minutiae **21.1** |

## 4. 1:N IDENTIFICATION — what changed

| Aspect | Before | After |
|---|--------|-------|
| Path handling | relative `"Exercise/biology_science/..."` (**broke** depending on CWD) | absolute `BASE_DIR` → runs from any folder |
| Stored template | `descriptors` | `(keypoints, descriptors)` tuple |
| FTA / quality gate | `len(desc) < 15` (crashed on tuple) | `num_keypoints(feat)` helper |
| Threshold | 1–2 (percent) | count-based: SIFT 13, ORB 10, Minutiae 10, LBP 50 |
| Multi-DB enroll | `build_mega_database` (SIFT only) | unchanged, paths fixed |

## 5. BUG FIXES (applied to all modules)

1. **Path bug** — `"Exercise/biology_science/fingerprints/DB1_B"` was wrong relative to the
   working directory → replaced with absolute `BASE_DIR`-based paths.
2. **SIFT app crash** (`apps/sift_app.py`, originally `Biology.py`) — menu options 1 & 3 used an **undefined** `folder_path` → `NameError`.
   Now `folder_path = DEFAULT_FOLDER` is defined.
3. **Performance** — `cv2.SIFT_create()` / `ORB_create()` were created **per image**; now created **once**.
4. **Language** — all comments and console UI translated **Vietnamese → English** (course is in English).

## 6. CROSS-DATABASE diagnostic — NEW (explains the "DB3_B problem")

SIFT + Combo 1, EER per database:

| DB | EER | genuine avg matches | Image mean brightness | Sensor type |
|----|-----|---------------------|-----------------------|-------------|
| DB1_B | 16.67% | 143 | 221 | optical |
| DB2_B | **6.19%** | 156 | 136 | optical (best quality) |
| DB3_B | 32.54% | 20 | **77 (very dark)** | capacitive |
| DB4_B | 48.25% | 4.8 | 146 | **synthetic (SFinGe)** |

**Why DB3_B is hard:** very dark, low-contrast capacitive images → SIFT finds far fewer stable
keypoints (~20 vs ~150). This is an **input-quality limitation**, not a code bug. DB4_B is even
harder because it is synthetically generated.

---

## 7. FINAL RANKING (for the Discussion section)

| Rank | Algorithm | EER (DB1_B) | One-line takeaway |
|------|-----------|-------------|-------------------|
| 1 | **SIFT + RANSAC** | **16.67%** | Best accuracy; rotation/scale invariant |
| 2 | Minutiae (CN) | 21.11% | Classic, great for visualization; sensitive to binarization |
| 3 | ORB + RANSAC | 21.90% | Faster, but binary descriptors weaker on ridges |
| 4 | LBP (grid) | ~44% | Global texture, no alignment → unsuitable for identity |

---

## 8. ITERATION TEMPLATE (copy this block for each future experiment)

```
### Iteration <N> — <date> — <short title>
- What changed: <preprocessing / algorithm / matching / parameter>
- Hypothesis: <why this might help>
- Setup: DB=<...>, ratio=<...>, threshold scan=<...>
- Result: EER SIFT=__ ORB=__ LBP=__ Minutiae=__  (genuine avg / imposter avg)
- Outcome vs previous: <better/worse/same>, by how much
- Why (interpretation): <...>
- Decision: <keep / revert / needs more testing>
```

### Ideas not yet tried (backlog)
- Full FVC protocol (all genuine pairs = 2800) for a more stable EER estimate.
- Multiple templates per finger in 1:N (use `_1.._3` instead of only `_1`).
- Liveness detection (dropped: no spoof dataset available).
- Score fusion (SIFT + Minutiae) to recover hard genuine pairs.

---

## 9. REPOSITORY STRUCTURE (refactor for GitHub)

To remove the duplication between the four demo apps, shared logic now lives in a
single `core/` package; experiments and apps both import from it.

```
core/          single source of truth (no duplication)
  io_utils.py      image reading, dataset paths, fixed random SEED
  preprocessing.py 6 combos: C1/C2/C3 and +Gabor variants
  features.py      SIFT / ORB / LBP / Minutiae extractors
  matching.py      match + RANSAC + scoring variants S1..S4
  evaluation.py    1:1 EER, 1:N Rank-1 / identification rate
apps/          interactive demos: sift_app / orb_app / lbp_app / minutiae_app
experiments/   run_experiments.py  (--exp 1|2|3|4)
results/       CSV outputs + figures/  (committed: part of the report)
fingerprints/  FVC2002 data (git-ignored: license-restricted)
```

Reproducibility: every experiment calls `core.io_utils.set_seed(42)` which seeds the
OpenCV RNG (used by RANSAC), NumPy and `random`. The seed is printed at run time.

---

## 10. EXPERIMENT PLAN (Comparative Study)

Designed as a **two-layer preprocessing study** plus algorithm and scoring studies.
Each experiment writes a CSV to `results/` and a figure to `results/figures/`, and a
filled Iteration block is appended below in §11.

- **Exp 1 — Preprocessing study (deep, SIFT only).** 6 combos (C1, C2, C3, C1+G, C2+G,
  C3+G) × 4 DBs, task 1:1. Per cell: EER, genuine avg, imposter avg, avg #keypoints,
  avg #RANSAC inliers (genuine). *Goal:* pick the best combo and explain why
  (denoise vs contrast vs keypoint stability); log the Gabor effect on keypoint count.
- **Exp 2 — Generalization check (all algos, 1 DB).** Same 6 combos × 4 algorithms on
  DB1_B. *Goal:* does the preprocessing conclusion transfer? Specifically test whether
  Gabor HELPS Minutiae (cleaner binarization → fewer spurious minutiae) while HURTING
  SIFT/ORB (fewer stable keypoints); log avg #minutiae before/after Gabor.
- **Exp 3 — Full algorithm comparison.** Production combo (C1), 4 algos × 4 DBs, BOTH
  tasks: 1:1 → EER; 1:N → Rank-1 accuracy + identification rate at the per-algorithm
  fixed threshold (no EER for 1:N). DB difficulty: DB2 easy, DB1 average, DB3 hard,
  DB4 synthetic/worst.
- **Exp 4 — Scoring strategy study (SIFT + ORB).** 4 scoring variants on DB1_B & DB3_B:
  S1 good-match count, S2 % matches, S3 RANSAC inliers (current), S4 inlier ratio.
  *Goal:* justify S1→S2→S3 and test whether S4 (geometry + size-normalization) beats S3.

Figures (`results/figures/`): EER bars per algo per DB (Exp 3), DET/ROC for 4 algos on
DB1_B, preprocessing-combo EER bars (Exp 1).

---

## 11. EXPERIMENT RESULTS (filled as each run completes)

> **Statistical caveat (applies to all EER numbers).** Each DB has only 70 genuine and
> 45 imposter 1:1 pairs, so EER is quantized: FAR changes in ~2.2% steps, FRR in ~1.4%
> steps. **EER differences smaller than ~3 percentage points are within noise** and must
> not be over-interpreted. Counts (keypoints, minutiae) are exact and reliable.

### Iteration 1 — 2026-06-11 — Exp 1: Preprocessing study (SIFT, 6 combos × 4 DBs)
- **What changed:** measured all 6 combos on SIFT for every DB (was: 4 combos, 2 DBs, no seed).
- **Setup:** SIFT, scoring S3 (RANSAC inliers), ratio 0.75, seed=42. CSV: `results/exp1_preprocessing_sift.csv`, figure: `results/figures/exp1_preprocessing_eer.png`.
- **MEASURED facts:**
  - DB difficulty dominates everything: DB2 (easy) **2.9–11%**, DB1 (avg) 14–26%, DB3 (hard)
    31–46%, DB4 (synthetic) **45–52%** — *regardless of combo*. Preprocessing cannot fix poor input.
  - Gabor **halves the SIFT keypoint count**: DB1 C1 2305 → C1+G 987; DB3 1968 → 682. (Exact, robust.)
  - Best EER per DB came from a *different* combo each time (DB1: C3+G 13.81; DB2: C3 2.86;
    DB3: C2 31.43; DB4: C2+G 45.08) — i.e. no combo dominates; spread is within noise.
- **Hypotheses (NOT proven):** Bilateral (C2) preserves ridge edges better than Gaussian, which
  *should* help on noisy DB3; data shows C2 ≤ C1 on DB1/DB3/DB4 but the gap is within noise.
- **Outcome vs previous:** **Corrects the earlier "Gabor hurts SIFT" claim.** That came from a
  single un-seeded 4-combo run. With a fixed seed and the full grid, Gabor's EER effect on SIFT
  is small and direction varies by DB (helped DB1, hurt DB3) — *not* a clean "hurts".
- **Decision:** keep **C1 (Gaussian+CLAHE)** as production default — consistently near-best,
  simplest, and keeps ~2× more keypoints than Gabor for the same accuracy. Report Gabor as a
  measured trade-off (halves keypoints, no consistent EER gain), not a winner.

### Iteration 2 — 2026-06-11 — Exp 2: Preprocessing generalization (6 combos × 4 algos, DB1_B)
- **What changed:** ran the same 6 combos on all 4 algorithms on DB1_B to test whether the
  preprocessing conclusion transfers. CSV: `results/exp2_preprocessing_all_algos.csv`, figure:
  `results/figures/exp2_generalization_eer.png`.
- **MEASURED facts:**
  - Best combo is **algorithm-dependent**: SIFT→C3+G (13.81%), ORB→C1 (21.90%),
    LBP→C2 (42.54%), Minutiae→C1+G (17.86%). The conclusion does **not** fully transfer.
  - Average Gabor effect (mean ΔEER over C1/C2/C3): SIFT −1.08pp, ORB −0.61pp, LBP −0.61pp,
    **Minutiae −2.46pp** (largest). All small; SIFT/ORB/LBP within noise, Minutiae the only
    one approaching significance.
  - **Test of the lecturer's hypothesis** ("Gabor → cleaner binarization → fewer spurious
    minutiae"): minutiae count barely moves with Gabor (C1 276.9 → C1+G 276.2; C2 275.4 →
    275.9; C3 237.5 → 271.8). → **Hypothesis REFUTED by measurement** — Gabor does not reduce
    spurious minutiae.
  - LBP stays **42–46%** for every combo — preprocessing cannot rescue a global-texture method.
- **Hypotheses (NOT proven):** Minutiae EER improved with Gabor *despite* unchanged minutiae
  count, so the gain likely comes from ridge-enhanced local SIFT descriptors at each minutia
  being more distinctive — not from fewer minutiae. Needs a dedicated test to confirm.
- **Outcome vs previous:** richer, more honest picture than §1–§2 above; supersedes the
  informal "C2 is best / Gabor doesn't help" note.
- **Decision:** keep **C1** as the shared production default (robust across all algorithms);
  mention per-algorithm best combos as a secondary finding. Do not adopt Gabor globally.

### Iteration 3 — 2026-06-11 — Exp 3: Full comparison (4 algos × 4 DBs, 1:1 + 1:N)
- **What changed:** benchmarked every algorithm on every DB with combo C1, for both tasks.
  CSVs: `results/exp3_algo_x_db_1to1.csv`, `exp3_algo_x_db_1toN.csv`. Figures:
  `exp3_eer_algo_x_db.png` (EER bars), `exp3_roc_db1.png` (ROC of 4 algos on DB1_B).
- **MEASURED facts (1:1 EER):** SIFT wins on DB2/DB1/DB3 (6.19 / 16.67 / 32.54); on synthetic
  DB4 every method is ~40–48% (ORB "best" at 40.16, but useless). DB difficulty ordering holds.
- **MEASURED facts (1:N Rank-1 / identification rate):** SIFT 91.4/88.6 (DB2), 80.0/72.9 (DB1),
  54.3/32.9 (DB3), 17.1/1.4 (DB4). LBP identification rate ≈ **0%** on every DB despite Rank-1
  24–50% (its scores never reach a usable threshold). Minutiae leads Rank-1 on hard DB3 (57.1%).
  DB4 near the 10% chance level for all methods.
- **Hypotheses (NOT proven):** Minutiae's edge on DB3 likely comes from explicit ridge-ending
  detection coping better with low-contrast capacitive images; lead is within sampling noise.
- **Outcome vs previous:** confirms ranking SIFT > Minutiae ≈ ORB > LBP from §7 with full 1:N data.
- **Decision:** recommend **SIFT** for the attendance demo; **LBP not usable for identification**
  (Rank-1 by luck, 0% identification rate). No method works on synthetic DB4.

### Iteration 3b — 2026-06-11 — Exp 3 extended with LATENCY (the 2nd primary criterion)
- **What changed:** the plan (Group_13) requires comparing algorithms on **EER + latency**. Added
  to Exp 3: extraction latency (ms/image), match latency (ms/comparison), 1:N query latency
  (extract + N×match, N=10), the 1:1 optimal threshold, plus an EER-vs-latency trade-off figure
  (`results/figures/exp3_eer_vs_latency.png`). New CSV columns in both Exp 3 CSVs.
- **MEASURED facts (mean over 4 DBs):** SIFT EER 25.91% / match 46.6 ms / extract 68 ms;
  ORB 26.94% / 15.6 ms / 11.8 ms; LBP 43.73% / **0.03 ms** / 29 ms; Minutiae 32.08% / 4.5 ms /
  **118 ms**. So SIFT = most accurate but slowest match; ORB = ~3× faster match + fastest enroll,
  small accuracy loss; LBP = fastest but useless; Minutiae = slowest enrollment (skeletonization).
- **Interpretation:** 1:N cost ≈ N×match, so at scale ORB's faster matching compounds (N=10000:
  ~156 s vs ~466 s for SIFT). Latency is hardware-dependent; relative ordering is the point.
- **Decision:** report **both** criteria. Recommend SIFT when accuracy-first / small gallery,
  ORB when gallery is large and match speed dominates.

### Iteration 4 — 2026-06-11 — Exp 4: Scoring strategy study (SIFT, ORB | DB1_B, DB3_B)
- **What changed:** compared 4 scoring variants (S1 good-count, S2 % matches, S3 RANSAC inliers,
  S4 inlier ratio) with RANSAC computed once per pair. CSV: `results/exp4_scoring.csv`.
- **MEASURED facts:** **S3 is best in all 4 cases** — SIFT DB1 16.67 (vs S1 20.0, S2 22.54,
  S4 24.37); SIFT DB3 32.54; ORB DB1 21.90; ORB DB3 32.78. **S4 is worst/near-worst.** For SIFT
  DB1 the impostor average jumps 5.47 (S3) → 32.70 (S4), collapsing the genuine/impostor gap.
- **Interpretation (supported by the impostor-avg numbers):** S4 discards evidence magnitude —
  an impostor with few but mostly-consistent matches gets an inflated ratio. Geometric
  verification matters, but the inlier *count* must be kept.
- **Decision:** keep **S3 (RANSAC inlier count)** as the production score. Negative result for S4
  recorded (a stated goal of the experiment).
