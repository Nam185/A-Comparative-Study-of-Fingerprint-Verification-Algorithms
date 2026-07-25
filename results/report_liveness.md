# Liveness Detection / Presentation Attack Detection — Report Text (Experiment 7)

*Ready-to-paste English text. Code: `core/liveness.py`, `experiments/run_experiments.py --exp 7`.
Outputs (once a dataset is provided): `results/exp7_liveness_pad.csv`,
`results/figures/exp7_liveness_roc.png`. Needs `scikit-learn` and a live/spoof dataset.*

---

## Experiment 7 — Fingerprint Liveness Detection (anti-spoofing)

### What we did and why
A matcher answers *"is this the enrolled person?"* — but it can be fooled by a **spoof**: a fake
finger made of gelatin, latex, silicone, wood glue or Play-Doh, a printed ridge pattern on tape, or
even a severed finger. **Presentation Attack Detection (PAD)**, or *liveness detection*, adds a second
question: *"is this a real, live finger at all?"* We add a PAD module so the system can reject fakes.

**Important paradigm difference.** The verification study (Experiments 1–6) is *zero-training* —
matching does not learn from data. PAD is fundamentally a **supervised binary classification** problem
(live vs spoof) and therefore **must be trained** on a labelled dataset. This is standard for all PAD
systems and is the honest nature of the task.

### Method (classical LivDet baseline)
We keep the traditional, explainable style of the project:

1. **Feature extraction** (`core/liveness.py`): a live finger and a replica differ in fine surface
   **texture**, ridge sharpness and perspiration pattern. We capture this with **multi-radius uniform
   Local Binary Patterns** (radii 1/2/3, i.e. 8/16/24 neighbours) concatenated into one histogram,
   plus two simple **image-quality statistics** (contrast and Laplacian sharpness) — 56 features total.
2. **Classifier**: a **Support Vector Machine** (RBF kernel) with standardised features, the classic
   LivDet software baseline.
3. **Protocol**: stratified train/test split (fixed seed); the model is trained on the training half
   and evaluated on the held-out test half.

### Metrics (ISO/IEC 30107-3 standard)
- **APCER** — Attack Presentation Classification Error Rate: fraction of **spoofs wrongly accepted as
  live** (the security-critical error).
- **BPCER** — Bona-fide Presentation Classification Error Rate: fraction of **live fingers wrongly
  rejected** (the convenience error).
- **ACER** = (APCER + BPCER) / 2, plus overall **accuracy** and **ROC AUC**.

### Dataset
PAD needs live **and** spoof images. The standard benchmark is **LivDet** (Liveness Detection
Competition — 2011/2013/2015/2017/2019/2021): thousands of live and fake fingerprints from several
sensors and spoof materials. It is free for academic research but requires a request/registration via
the official site (University of Cagliari); some mirrors exist on Kaggle/GitHub. Our loader
(`load_live_spoof`) **auto-detects** the class from folder names, so LivDet layouts
(`.../Live/...`, `.../Fake/Gelatine/...`) or a simple `Live/` + `Spoof/` split both work — place the
data under `fingerprints/liveness/`.

### How to inspect it (visual proof, not just numbers)
- `python experiments/visualize_liveness.py <image>` renders **what the detector extracts**: the
  multi-radius LBP texture maps and the LBP histogram (Figure *liveness_features_single*). Pass a
  live and a spoof image to see the texture/feature difference side by side.
- Running `--exp 7` on a real dataset additionally saves a **ROC curve**, a **confusion matrix**, and
  a **sample-predictions** panel (each test image with its predicted label and confidence, green =
  correct / red = wrong) — so the classifier's decisions can be checked by eye.

### Results
*(To be filled after running on a real dataset: `python experiments/run_experiments.py --exp 7`.)*
For reference, classical LBP+SVM PAD in the literature reaches roughly **85–95 % accuracy** on
same-sensor, same-material LivDet splits; accuracy drops on **unseen spoof materials or sensors**,
which is the genuinely hard, open problem in PAD (deep-learning methods currently lead there).

### Limitations and future work *(optional)*
- The hardest case is **cross-material / cross-sensor generalization** (a spoof made from a material
  not seen in training); a fair evaluation should use LivDet's official train/test protocol.
- Stronger classical texture features (BSIF, WLD) or a small CNN would likely improve APCER; combining
  PAD with the matcher yields a full "is it the right person **and** a real finger?" system.
