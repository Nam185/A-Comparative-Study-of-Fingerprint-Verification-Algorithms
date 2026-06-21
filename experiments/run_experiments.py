"""Single entry point to reproduce every experiment.

Usage:
    python experiments/run_experiments.py --exp 1      # preprocessing study (SIFT, 4 DBs)
    python experiments/run_experiments.py --exp 2      # preprocessing generalization (4 algos, DB1_B)
    python experiments/run_experiments.py --exp 3      # full algo x DB comparison (1:1 + 1:N)  [TODO]
    python experiments/run_experiments.py --exp 4      # scoring strategy study (SIFT/ORB)        [TODO]
    python experiments/run_experiments.py --all
No flag -> interactive menu.

Every experiment fixes the RANSAC seed, writes a CSV to results/, saves a figure
to results/figures/, and prints a summary you can screenshot for the report.
"""
import os
import sys
import csv
import glob
import re
import argparse
from time import perf_counter
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # save figures to file, no display needed
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.io_utils import BASE_DIR, DBS, DB_DIFFICULTY, set_seed, read_gray, img_path
from core.preprocessing import COMBOS, COMBO_DESC
from core.features import EXTRACTORS
from core.matching import match, match_components, score_from_components, SCORING_VARIANTS
from core.evaluation import collect_scores, eer, rank1_and_idrate, FINGERS, IMPRESSIONS
from core import minutiae_native as MN
from core.features import sift_features

RESULTS = os.path.join(BASE_DIR, "results")
FIGURES = os.path.join(RESULTS, "figures")
os.makedirs(FIGURES, exist_ok=True)

COMBO_ORDER = ["C1", "C2", "C3", "C1+G", "C2+G", "C3+G"]
ALGO_ORDER = ["SIFT", "ORB", "LBP", "Minutiae"]
# Per-algorithm scoring + 1:N fixed threshold (from production defaults)
ALGO_SCORING = {"SIFT": "S3", "ORB": "S3", "LBP": "chi", "Minutiae": "S3"}
ALGO_THRESHOLD = {"SIFT": 13, "ORB": 10, "LBP": 50, "Minutiae": 10}


def _build_feat_fn(db, combo_fn, extractor):
    """Return a cached feat(finger_id, impression) for one (db, combo, algo)."""
    cache = {}

    def feat(fid, imp):
        k = (fid, imp)
        if k not in cache:
            cache[k] = extractor(combo_fn(read_gray(img_path(db, fid, imp))))
        return cache[k]

    return feat, cache


def _eval_1to1(db, combo_fn, extractor, ratio=0.75, scoring="S3"):
    """Run 1:1 protocol for one (db, combo, algo). Returns a metrics dict."""
    cache = {}

    def feat(fid, imp):
        k = (fid, imp)
        if k not in cache:
            img = read_gray(img_path(db, fid, imp))
            cache[k] = extractor(combo_fn(img))
        return cache[k]

    use_scoring = "S3" if scoring == "chi" else scoring
    match_fn = lambda a, b: match(a, b, ratio=ratio, scoring=use_scoring)
    genuine, imposter = collect_scores(feat, match_fn)
    e, thr = eer(genuine, imposter)

    feats = list(cache.values())
    n_kps = [f.get("n_kp", 0) for f in feats]
    n_min = [f.get("n_minutiae") for f in feats if f.get("n_minutiae") is not None]
    return {
        "eer": e, "threshold": thr,
        "gen_avg": float(np.mean(genuine)), "imp_avg": float(np.mean(imposter)),
        "avg_keypoints": float(np.mean(n_kps)) if n_kps else 0.0,
        "avg_inliers_genuine": float(np.mean(genuine)),  # == inliers when scoring S3
        "avg_minutiae": float(np.mean(n_min)) if n_min else None,
        "genuine": genuine, "imposter": imposter,
    }


# ============================== EXPERIMENT 1 ==============================
def experiment_1():
    seed = set_seed()
    print(f"\n=== EXPERIMENT 1: Preprocessing study (SIFT, 6 combos x 4 DBs) | seed={seed} ===\n")
    rows = []
    eer_matrix = {db: {} for db in DBS}
    print(f"{'DB':7s} {'combo':6s} {'EER%':>7s} {'genAvg':>8s} {'impAvg':>7s} {'avgKP':>7s} {'avgInl':>7s}")
    for db in DBS:
        for combo in COMBO_ORDER:
            m = _eval_1to1(db, COMBOS[combo], EXTRACTORS["SIFT"])
            eer_matrix[db][combo] = m["eer"]
            print(f"{db:7s} {combo:6s} {m['eer']:7.2f} {m['gen_avg']:8.1f} "
                  f"{m['imp_avg']:7.2f} {m['avg_keypoints']:7.0f} {m['avg_inliers_genuine']:7.1f}")
            rows.append({"db": db, "difficulty": DB_DIFFICULTY[db], "combo": combo,
                         "combo_desc": COMBO_DESC[combo], "eer_pct": round(m["eer"], 2),
                         "genuine_avg": round(m["gen_avg"], 2), "imposter_avg": round(m["imp_avg"], 2),
                         "avg_keypoints": round(m["avg_keypoints"], 1),
                         "avg_inliers_genuine": round(m["avg_inliers_genuine"], 1)})

    csv_path = os.path.join(RESULTS, "exp1_preprocessing_sift.csv")
    _write_csv(csv_path, rows)
    fig_path = os.path.join(FIGURES, "exp1_preprocessing_eer.png")
    _grouped_bar(eer_matrix, DBS, COMBO_ORDER, "EER (%)",
                 "Exp 1: SIFT EER per preprocessing combo across DBs", fig_path,
                 group_by="db")
    print(f"\nSaved: {csv_path}\nSaved: {fig_path}")
    return rows


# ============================== EXPERIMENT 2 ==============================
def experiment_2(db="DB1_B"):
    seed = set_seed()
    print(f"\n=== EXPERIMENT 2: Preprocessing generalization ({db}, 6 combos x 4 algos) | seed={seed} ===\n")
    rows = []
    eer_matrix = {algo: {} for algo in ALGO_ORDER}
    print(f"{'algo':10s} {'combo':6s} {'EER%':>7s} {'genAvg':>8s} {'impAvg':>8s} {'avgKP/min':>10s}")
    for algo in ALGO_ORDER:
        for combo in COMBO_ORDER:
            m = _eval_1to1(db, COMBOS[combo], EXTRACTORS[algo], scoring=ALGO_SCORING[algo])
            eer_matrix[algo][combo] = m["eer"]
            extra = m["avg_minutiae"] if m["avg_minutiae"] is not None else m["avg_keypoints"]
            print(f"{algo:10s} {combo:6s} {m['eer']:7.2f} {m['gen_avg']:8.2f} "
                  f"{m['imp_avg']:8.2f} {extra:10.1f}")
            rows.append({"db": db, "algo": algo, "combo": combo,
                         "combo_desc": COMBO_DESC[combo], "eer_pct": round(m["eer"], 2),
                         "genuine_avg": round(m["gen_avg"], 2), "imposter_avg": round(m["imp_avg"], 2),
                         "avg_keypoints": round(m["avg_keypoints"], 1),
                         "avg_minutiae": round(m["avg_minutiae"], 1) if m["avg_minutiae"] is not None else ""})

    csv_path = os.path.join(RESULTS, "exp2_preprocessing_all_algos.csv")
    _write_csv(csv_path, rows)
    fig_path = os.path.join(FIGURES, "exp2_generalization_eer.png")
    _grouped_bar(eer_matrix, ALGO_ORDER, COMBO_ORDER, "EER (%)",
                 f"Exp 2: EER per combo across algorithms ({db})", fig_path, group_by="algo")
    print(f"\nSaved: {csv_path}\nSaved: {fig_path}")

    # Per-algorithm verdict: best combo + Gabor effect
    print("\n--- Per-algorithm verdict (best combo, Gabor effect) ---")
    for algo in ALGO_ORDER:
        em = eer_matrix[algo]
        best = min(em, key=em.get)
        g_eff = np.mean([em[f"{c}+G"] - em[c] for c in ["C1", "C2", "C3"]])
        sign = "HURTS" if g_eff > 0 else "HELPS"
        print(f"{algo:10s} best={best} ({em[best]:.2f}%)  Gabor avg dEER={g_eff:+.2f}pp -> {sign}")
    return rows


# ============================== helpers ==============================
def _write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _grouped_bar(matrix, groups, series, ylabel, title, path, group_by="db"):
    """matrix[group][serie] = value. Bars grouped by `groups`, colored by `series`."""
    x = np.arange(len(groups))
    width = 0.8 / len(series)
    plt.figure(figsize=(12, 6))
    for i, s in enumerate(series):
        vals = [matrix[g].get(s, 0) for g in groups]
        plt.bar(x + i * width, vals, width, label=s, edgecolor="black", linewidth=0.4)
    plt.xticks(x + width * (len(series) - 1) / 2, groups)
    plt.ylabel(ylabel, fontweight="bold")
    plt.title(title, fontweight="bold")
    plt.legend(title="combo", ncol=len(series))
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _roc_curve(curves, path, title):
    """curves[algo] = (genuine, imposter). Plot FAR (x) vs TAR=1-FRR (y)."""
    plt.figure(figsize=(8, 7))
    for algo in ALGO_ORDER:
        if algo not in curves:
            continue
        g = np.asarray(curves[algo][0], float)
        im = np.asarray(curves[algo][1], float)
        ts = np.unique(np.concatenate([g, im]))
        far = [np.mean(im >= t) for t in ts]
        tar = [np.mean(g >= t) for t in ts]   # genuine acceptance = 1 - FRR
        order = np.argsort(far)
        plt.plot(np.array(far)[order], np.array(tar)[order], marker=".", label=algo)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4, label="chance")
    plt.xlabel("False Accept Rate (FAR)", fontweight="bold")
    plt.ylabel("True Accept Rate (1 - FRR)", fontweight="bold")
    plt.title(title, fontweight="bold")
    plt.legend()
    plt.grid(linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _tradeoff_plot(agg, path):
    """Accuracy vs speed: x = mean match latency (log), y = mean EER. Lower-left is better."""
    plt.figure(figsize=(8, 6))
    for algo in ALGO_ORDER:
        x = float(np.mean(agg[algo]["match_ms"]))
        y = float(np.mean(agg[algo]["eer"]))
        plt.scatter(x, y, s=140, edgecolor="black", zorder=3)
        plt.annotate(algo, (x, y), textcoords="offset points", xytext=(9, 5), fontweight="bold")
    plt.xscale("log")
    plt.xlabel("Match latency (ms per comparison, log scale) - lower is faster", fontweight="bold")
    plt.ylabel("Mean EER over 4 DBs (%) - lower is more accurate", fontweight="bold")
    plt.title("Exp 3: Accuracy vs Speed trade-off (best = lower-left)", fontweight="bold")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ============================== EXPERIMENT 3 ==============================
def experiment_3():
    seed = set_seed()
    print(f"\n=== EXPERIMENT 3: Full comparison (combo C1, 4 algos x 4 DBs, 1:1 + 1:N) | seed={seed} ===")
    print("Primary criteria: EER (accuracy) and Latency (speed). N=10 enrolled templates.\n")
    rows_11, rows_1n = [], []
    eer_matrix = {db: {} for db in DBS}
    curves = {}                       # DB1_B genuine/imposter scores for the ROC figure
    agg = {a: {"eer": [], "match_ms": [], "extract_ms": []} for a in ALGO_ORDER}

    print(f"{'DB':7s} {'algo':10s} | {'EER%':>6s} {'optThr':>7s} | {'Rank1%':>7s} {'IDrate%':>8s} | "
          f"{'extr.ms':>8s} {'match.ms':>9s} {'1:N ms':>8s}")
    N_TEMPLATES = 10
    for db in DBS:
        for algo in ALGO_ORDER:
            combo_fn, extractor = COMBOS["C1"], EXTRACTORS[algo]
            cache, extract_times = {}, []

            def feat(fid, imp):
                k = (fid, imp)
                if k not in cache:
                    img = read_gray(img_path(db, fid, imp))   # disk read NOT timed
                    t0 = perf_counter()
                    cache[k] = extractor(combo_fn(img))        # preprocess + extract IS timed
                    extract_times.append(perf_counter() - t0)
                return cache[k]

            match_times = []

            def tmatch(a, b):
                t0 = perf_counter()
                s = match(a, b, scoring="S3")                  # S3 for kp; LBP dispatches to chi
                match_times.append(perf_counter() - t0)
                return s

            # --- 1:1: EER + optimal threshold ---
            gen, imp = collect_scores(feat, tmatch)
            e, opt_thr = eer(gen, imp)
            eer_matrix[db][algo] = e
            if db == "DB1_B":
                curves[algo] = (gen, imp)

            # --- 1:N: Rank-1 + identification rate at the fixed operating threshold ---
            thr = ALGO_THRESHOLD[algo]
            rank1, idrate, total = rank1_and_idrate(feat, lambda a, b: match(a, b, scoring="S3"), thr)

            # --- Latency ---
            extract_ms = float(np.mean(extract_times)) * 1000
            match_ms = float(np.mean(match_times)) * 1000
            ident_ms = extract_ms + N_TEMPLATES * match_ms      # cost of one 1:N query (N=10)
            agg[algo]["eer"].append(e)
            agg[algo]["match_ms"].append(match_ms)
            agg[algo]["extract_ms"].append(extract_ms)

            print(f"{db:7s} {algo:10s} | {e:6.2f} {opt_thr:7.1f} | {rank1:7.1f} {idrate:8.1f} | "
                  f"{extract_ms:8.2f} {match_ms:9.3f} {ident_ms:8.2f}")
            rows_11.append({"db": db, "difficulty": DB_DIFFICULTY[db], "algo": algo,
                            "eer_pct": round(e, 2), "optimal_threshold": round(opt_thr, 2),
                            "genuine_avg": round(float(np.mean(gen)), 2),
                            "imposter_avg": round(float(np.mean(imp)), 2),
                            "extract_ms_per_img": round(extract_ms, 2),
                            "match_ms_per_cmp": round(match_ms, 3)})
            rows_1n.append({"db": db, "difficulty": DB_DIFFICULTY[db], "algo": algo,
                            "threshold": thr, "rank1_acc_pct": round(rank1, 1),
                            "identification_rate_pct": round(idrate, 1), "n_queries": total,
                            "ident_latency_ms_per_query": round(ident_ms, 2)})

    _write_csv(os.path.join(RESULTS, "exp3_algo_x_db_1to1.csv"), rows_11)
    _write_csv(os.path.join(RESULTS, "exp3_algo_x_db_1toN.csv"), rows_1n)
    fig1 = os.path.join(FIGURES, "exp3_eer_algo_x_db.png")
    _grouped_bar(eer_matrix, DBS, ALGO_ORDER, "EER (%)",
                 "Exp 3: 1:1 EER per algorithm across databases (combo C1)", fig1, group_by="db")
    fig2 = os.path.join(FIGURES, "exp3_roc_db1.png")
    _roc_curve(curves, fig2, "Exp 3: ROC curve of the 4 algorithms (DB1_B)")
    fig3 = os.path.join(FIGURES, "exp3_eer_vs_latency.png")
    _tradeoff_plot(agg, fig3)
    print(f"\nSaved: exp3_algo_x_db_1to1.csv, exp3_algo_x_db_1toN.csv")
    print(f"Saved: {fig1}\nSaved: {fig2}\nSaved: {fig3}")

    # Accuracy/speed summary averaged over the four DBs
    print("\n--- Mean over 4 DBs (primary criteria) ---")
    print(f"{'algo':10s} {'mean EER%':>9s} {'match ms/cmp':>13s} {'extract ms/img':>15s}")
    for algo in ALGO_ORDER:
        print(f"{algo:10s} {np.mean(agg[algo]['eer']):9.2f} "
              f"{np.mean(agg[algo]['match_ms']):13.3f} {np.mean(agg[algo]['extract_ms']):15.2f}")
    return rows_11, rows_1n


# ============================== EXPERIMENT 4 ==============================
def experiment_4():
    seed = set_seed()
    print(f"\n=== EXPERIMENT 4: Scoring strategy study (SIFT, ORB | DB1_B, DB3_B) | seed={seed} ===\n")
    rows = []
    print(f"{'algo':6s} {'DB':7s} {'variant':8s} {'EER%':>7s} {'genAvg':>9s} {'impAvg':>9s}  desc")
    for algo in ["SIFT", "ORB"]:
        for db in ["DB1_B", "DB3_B"]:
            feat, _ = _build_feat_fn(db, COMBOS["C1"], EXTRACTORS[algo])
            # Compute RANSAC components ONCE per pair, then derive every scoring variant
            gen_comp, imp_comp = [], []
            for fid in FINGERS:
                base = feat(fid, 1)
                for imp in IMPRESSIONS:
                    gen_comp.append(match_components(base, feat(fid, imp)))
            for i in FINGERS:
                a = feat(i, 1)
                for j in FINGERS:
                    if j > i:
                        imp_comp.append(match_components(a, feat(j, 1)))
            for variant in ["S1", "S2", "S3", "S4"]:
                g = [score_from_components(c, variant) for c in gen_comp]
                m = [score_from_components(c, variant) for c in imp_comp]
                e, _ = eer(g, m)
                print(f"{algo:6s} {db:7s} {variant:8s} {e:7.2f} {np.mean(g):9.2f} "
                      f"{np.mean(m):9.2f}  {SCORING_VARIANTS[variant]}")
                rows.append({"algo": algo, "db": db, "variant": variant,
                             "variant_desc": SCORING_VARIANTS[variant], "eer_pct": round(e, 2),
                             "genuine_avg": round(float(np.mean(g)), 2),
                             "imposter_avg": round(float(np.mean(m)), 2)})
    _write_csv(os.path.join(RESULTS, "exp4_scoring.csv"), rows)
    print(f"\nSaved: {os.path.join(RESULTS, 'exp4_scoring.csv')}")
    return rows


# ============================== EXPERIMENT 5 (SOCOFing, large real 1:N) ==============================
SOCO_REAL = os.path.join(BASE_DIR, "fingerprints", "SOCOFing", "Real")
_SOCO_ALT = os.path.join(BASE_DIR, "fingerprints", "SOCOFing", "Altered")
SOCO_ALT = {d: os.path.join(_SOCO_ALT, f"Altered-{d}") for d in ["Easy", "Medium", "Hard"]}
# Candidate preprocessing pipelines tuned PER METHOD on SOCOFing (label, upscale, combo)
SOCO_CANDIDATES = [("x2+C1", 2, "C1"), ("x3+C1", 3, "C1"), ("x3+C1G", 3, "C1+G"), ("x3+C2", 3, "C2")]


def _soco_id2alt(difficulty):
    """Map identity -> one altered image path for the given difficulty level."""
    out = {}
    for a in glob.glob(os.path.join(SOCO_ALT[difficulty], "*.BMP")):
        idt = _soco_identity(a)
        if idt not in out:
            out[idt] = a
    return out


def _soco_identity(path):
    """'100__M_Left_index_finger_CR.BMP' -> '100__M_Left_index_finger' (strip alteration tag)."""
    return re.sub(r"_(CR|Obl|Zcut)$", "", os.path.basename(path)[:-4])


def _soco_real_path(identity):
    return os.path.join(SOCO_REAL, identity + ".BMP")


def _soco_extract(path, method, cfg):
    """cfg = (label, upscale, combo_key). Upscale the tiny image, then the method's combo."""
    _, upscale, combo_key = cfg
    img = cv2.resize(read_gray(path), None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    enh = COMBOS[combo_key](img)
    return sift_features(enh) if method == "SIFT" else MN.minutiae_features(enh)


def _soco_matchfn(method):
    return (lambda a, b: match(a, b, scoring="S3")) if method == "SIFT" else MN.match


def experiment_5(tune_n=80, gallery=100, queries=100):
    """SOCOFing SIFT vs Minutiae on a large REAL 1:N gallery.
    (B) tune each method's own preprocessing, (A) latency + scaling,
    (C) 1:N accuracy across alteration difficulty (Easy/Medium/Hard).
    """
    seed = set_seed()
    real = sorted(glob.glob(os.path.join(SOCO_REAL, "*.BMP")))
    if not real:
        print("SOCOFing Real not found under fingerprints/SOCOFing/Real. Skipping Exp 5.")
        return
    # identity -> altered path, per difficulty
    alt_by_diff = {d: _soco_id2alt(d) for d in ["Easy", "Medium", "Hard"]}
    valid_ids = sorted(i for i in alt_by_diff["Hard"] if os.path.exists(_soco_real_path(i))
                       and i in alt_by_diff["Easy"] and i in alt_by_diff["Medium"])
    print(f"\n=== EXPERIMENT 5: SOCOFing SIFT vs Minutiae (real, large 1:N) | seed={seed} ===")
    print(f"Real gallery available: {len(real)} | usable identities (Real+Easy+Med+Hard): {len(valid_ids)}\n")

    matchfn = {m: _soco_matchfn(m) for m in ["SIFT", "Minutiae"]}

    # ---------- (B) Per-method preprocessing tuning (on the HARD set, where it matters) ----------
    print("--- (B) Preprocessing tuning per method (EER on Altered-Hard validation subset) ---")
    print(f"{'method':10s} {'candidate':8s} {'EER%':>7s} {'genAvg':>8s} {'impAvg':>8s}")
    best_cfg, best_thr = {}, {}
    tune_ids = valid_ids[:tune_n]
    hard = alt_by_diff["Hard"]
    tune_rows = []
    for method in ["SIFT", "Minutiae"]:
        results = []
        for cfg in SOCO_CANDIDATES:
            realf = {i: _soco_extract(_soco_real_path(i), method, cfg) for i in tune_ids}
            altf = {i: _soco_extract(hard[i], method, cfg) for i in tune_ids}
            gen = [matchfn[method](realf[i], altf[i]) for i in tune_ids]
            imp = [matchfn[method](altf[i], realf[tune_ids[(k + 1) % len(tune_ids)]])
                   for k, i in enumerate(tune_ids)]
            e, thr = eer(gen, imp)
            results.append((e, thr, cfg))
            print(f"{method:10s} {cfg[0]:8s} {e:7.2f} {np.mean(gen):8.1f} {np.mean(imp):8.1f}")
            tune_rows.append({"method": method, "candidate": cfg[0], "eer_pct": round(e, 2),
                              "genuine_avg": round(float(np.mean(gen)), 2),
                              "imposter_avg": round(float(np.mean(imp)), 2)})
        results.sort(key=lambda r: (r[0], r[2][1]))   # tie-break: lower EER, then cheaper upscale
        best_cfg[method], best_thr[method] = results[0][2], results[0][1]
        print(f"  -> {method} best preprocessing: {best_cfg[method][0]} "
              f"(EER {results[0][0]:.2f}%, threshold {best_thr[method]:.1f})\n")
    _write_csv(os.path.join(RESULTS, "exp5_preprocessing_tuning.csv"), tune_rows)

    # ---------- (A) Latency with each method's tuned preprocessing ----------
    print("--- (A) Latency (each method at its tuned best preprocessing) ---")
    print(f"{'method':10s} {'avg feats':>10s} {'extract ms':>11s} {'match ms':>9s}")
    timings, lat_rows = {}, []
    speed_paths = real[:120]
    for method in ["SIFT", "Minutiae"]:
        cfg = best_cfg[method]
        feats, ext_ms = [], []
        for p in speed_paths:
            t0 = perf_counter(); f = _soco_extract(p, method, cfg); ext_ms.append((perf_counter()-t0)*1000)
            feats.append(f)
        mfn = matchfn[method]
        match_ms = []
        for i in range(len(feats) - 1):
            t0 = perf_counter(); mfn(feats[i], feats[i+1]); match_ms.append((perf_counter()-t0)*1000)
        cnt = np.mean([f["n_kp"] if method == "SIFT" else f["n_minutiae"] for f in feats])
        em, mm = float(np.mean(ext_ms)), float(np.mean(match_ms))
        timings[method] = (em, mm)
        print(f"{method:10s} {cnt:10.0f} {em:11.2f} {mm:9.3f}")
        lat_rows.append({"algorithm": method, "preprocessing": cfg[0], "avg_features": round(cnt, 0),
                         "extract_ms_per_img": round(em, 2), "match_ms_per_cmp": round(mm, 3)})
    _write_csv(os.path.join(RESULTS, "exp5_socofing_latency.csv"), lat_rows)

    Ns = [100, 1000, 6000, 60000]
    proj = {m: [(timings[m][0] + N * timings[m][1]) / 1000.0 for N in Ns] for m in timings}
    print(f"\n  1:N identification time per query (extract + N x match):")
    print("  " + f"{'N':>8s} {'SIFT(s)':>10s} {'Minutiae(s)':>12s}")
    for k, N in enumerate(Ns):
        print("  " + f"{N:>8d} {proj['SIFT'][k]:>10.2f} {proj['Minutiae'][k]:>12.2f}")
    plt.figure(figsize=(8, 6))
    for m in ["SIFT", "Minutiae"]:
        plt.plot(Ns, proj[m], marker="o", label=m)
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("Gallery size N (log)", fontweight="bold")
    plt.ylabel("Identification time per query (s, log)", fontweight="bold")
    plt.title("Exp 5: 1:N identification cost vs gallery size (SOCOFing)", fontweight="bold")
    plt.legend(); plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, "exp5_latency_scaling.png"), dpi=150); plt.close()

    # ---------- (C) 1:N accuracy across alteration difficulty ----------
    print(f"\n--- (C) 1:N accuracy: enroll {gallery} Real, query {queries} altered (Rank-1 / IDrate %) ---")
    gal_ids = valid_ids[:gallery]
    q_ids = valid_ids[:queries]
    print(f"{'method':10s} {'difficulty':10s} {'Rank1%':>7s} {'IDrate%':>8s}")
    acc_rows = []
    acc_matrix = {m: {} for m in ["SIFT", "Minutiae"]}
    for method in ["SIFT", "Minutiae"]:
        cfg, thr = best_cfg[method], best_thr[method]
        gallery_feats = {i: _soco_extract(_soco_real_path(i), method, cfg) for i in gal_ids}
        mfn = matchfn[method]
        for diff in ["Easy", "Medium", "Hard"]:
            id2a = alt_by_diff[diff]
            correct = correct_thr = 0
            for tid in q_ids:
                qf = _soco_extract(id2a[tid], method, cfg)
                best_id, best_s = None, -1.0
                for gid, gf in gallery_feats.items():
                    s = mfn(qf, gf)
                    if s > best_s:
                        best_s, best_id = s, gid
                if best_id == tid:
                    correct += 1
                    if best_s >= thr:
                        correct_thr += 1
            rank1 = correct / len(q_ids) * 100
            idrate = correct_thr / len(q_ids) * 100
            acc_matrix[method][diff] = rank1
            print(f"{method:10s} {diff:10s} {rank1:7.1f} {idrate:8.1f}")
            acc_rows.append({"method": method, "preprocessing": cfg[0], "difficulty": diff,
                             "gallery": len(gal_ids), "queries": len(q_ids), "threshold": round(thr, 1),
                             "rank1_acc_pct": round(rank1, 1), "identification_rate_pct": round(idrate, 1)})
    _write_csv(os.path.join(RESULTS, "exp5_socofing_accuracy.csv"), acc_rows)

    # Accuracy-vs-difficulty figure
    plt.figure(figsize=(8, 6))
    diffs = ["Easy", "Medium", "Hard"]
    for method in ["SIFT", "Minutiae"]:
        plt.plot(diffs, [acc_matrix[method][d] for d in diffs], marker="o", label=method)
    plt.ylabel("Rank-1 accuracy (%)", fontweight="bold")
    plt.xlabel("Alteration difficulty", fontweight="bold")
    plt.title(f"Exp 5: Rank-1 vs alteration difficulty (gallery={gallery}, SOCOFing)", fontweight="bold")
    plt.ylim(0, 105); plt.legend(); plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, "exp5_accuracy_vs_difficulty.png"), dpi=150); plt.close()

    print("\nSaved: exp5_preprocessing_tuning.csv, exp5_socofing_latency.csv, exp5_socofing_accuracy.csv")
    print("Saved: results/figures/exp5_latency_scaling.png, exp5_accuracy_vs_difficulty.png")
    return acc_rows


# ============================== EXPERIMENT 6 (best-vs-best accuracy on FVC B) ==============================
def experiment_6():
    """Best-vs-best ACCURACY: SIFT vs the DEDICATED Minutiae matcher (theta + geometry,
    not SIFT descriptors) on FVC2002 set B (small but hard, real multi-impression data).
    Complements Exp 5 (speed on the large SOCOFing gallery)."""
    seed = set_seed()
    print(f"\n=== EXPERIMENT 6: Best-vs-best accuracy on FVC B (SIFT vs Minutiae-native) | seed={seed} ===")
    print("FVC B has multiple REAL impressions per finger and hard imagery -> EER is meaningful here.\n")

    # (B) pick the dedicated matcher's own best preprocessing on DB1_B (1:1 EER)
    print("--- Minutiae-native preprocessing selection on DB1_B ---")
    print(f"{'combo':6s} {'EER%':>7s} {'genAvg':>8s} {'impAvg':>8s}")
    best_min, best_e = "C1", 100.0
    for combo in ["C1", "C1+G", "C2"]:
        feat, _ = _build_feat_fn("DB1_B", COMBOS[combo], MN.minutiae_features)
        gen, imp = collect_scores(feat, MN.match)
        e, _ = eer(gen, imp)
        print(f"{combo:6s} {e:7.2f} {np.mean(gen):8.1f} {np.mean(imp):8.1f}")
        if e < best_e:
            best_e, best_min = e, combo
    print(f"  -> Minutiae-native best preprocessing: {best_min}\n")

    # (A) per-DB 1:1 EER, SIFT (C1) vs Minutiae-native (best_min)
    methods = [("SIFT", COMBOS["C1"], EXTRACTORS["SIFT"], lambda a, b: match(a, b, scoring="S3")),
               ("Minutiae-native", COMBOS[best_min], MN.minutiae_features, MN.match)]
    eer_matrix = {db: {} for db in DBS}
    rows = []
    print(f"{'DB':7s} {'method':16s} {'EER%':>7s} {'optThr':>7s} {'genAvg':>8s} {'impAvg':>8s}")
    for db in DBS:
        for name, combo_fn, extractor, mfn in methods:
            feat, _ = _build_feat_fn(db, combo_fn, extractor)
            gen, imp = collect_scores(feat, mfn)
            e, thr = eer(gen, imp)
            eer_matrix[db][name] = e
            print(f"{db:7s} {name:16s} {e:7.2f} {thr:7.1f} {np.mean(gen):8.1f} {np.mean(imp):8.1f}")
            rows.append({"db": db, "difficulty": DB_DIFFICULTY[db], "method": name,
                         "eer_pct": round(e, 2), "optimal_threshold": round(float(thr), 2),
                         "genuine_avg": round(float(np.mean(gen)), 2),
                         "imposter_avg": round(float(np.mean(imp)), 2)})

    _write_csv(os.path.join(RESULTS, "exp6_accuracy_fvcB.csv"), rows)
    fig = os.path.join(FIGURES, "exp6_accuracy_fvcB.png")
    _grouped_bar(eer_matrix, DBS, ["SIFT", "Minutiae-native"], "EER (%)",
                 "Exp 6: 1:1 EER on FVC B - SIFT vs dedicated Minutiae matcher", fig, group_by="db")
    print(f"\nMean EER over 4 DBs: SIFT {np.mean([eer_matrix[d]['SIFT'] for d in DBS]):.2f}%  |  "
          f"Minutiae-native {np.mean([eer_matrix[d]['Minutiae-native'] for d in DBS]):.2f}%")
    print(f"Saved: results/exp6_accuracy_fvcB.csv\nSaved: {fig}")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=int, choices=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    runners = {1: experiment_1, 2: experiment_2, 3: experiment_3, 4: experiment_4,
               5: experiment_5, 6: experiment_6}
    if args.all:
        for i in [1, 2, 3, 4, 5, 6]:
            runners[i]()
    elif args.exp:
        runners[args.exp]()
    else:
        print("Select: 1=Preprocessing 2=Generalization 3=Algo x DB 4=Scoring "
              "5=SOCOFing speed 6=FVC accuracy(SIFT vs Minutiae-native)")
        choice = input("Experiment number: ").strip()
        runners.get(int(choice), lambda: print("Invalid"))() if choice.isdigit() else print("Invalid")


if __name__ == "__main__":
    main()
