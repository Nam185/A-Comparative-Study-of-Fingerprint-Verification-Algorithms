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


# ============================== EXPERIMENT 5 (SOCOFing, large 1:N) ==============================
SOCO_DIR = os.path.join(BASE_DIR, "fingerprints")
SOCO_UPSCALE = 3  # SOCOFing images are tiny (~96x103 px) -> upscale before processing


def _soco_prep(path, method):
    """Per-method best preprocessing for SOCOFing (upscale + the method's own combo)."""
    img = read_gray(path)
    img = cv2.resize(img, None, fx=SOCO_UPSCALE, fy=SOCO_UPSCALE, interpolation=cv2.INTER_CUBIC)
    if method == "SIFT":
        return COMBOS["C1"](img)        # SIFT's best combo from the FVC study
    return COMBOS["C1+G"](img)          # Minutiae's best combo (Gabor) from the FVC study


def _soco_feature(path, method):
    enh = _soco_prep(path, method)
    return sift_features(enh) if method == "SIFT" else MN.minutiae_features(enh)


def _soco_match(method):
    return (lambda a, b: match(a, b, scoring="S3")) if method == "SIFT" else MN.match


def experiment_5(sample=120):
    """SOCOFing: SIFT vs Minutiae for a LARGE 1:N gallery. Speed-at-scale is the focus.

    Accuracy needs genuine pairs (Real vs Altered); if the Altered folder is absent the
    accuracy part is skipped and only the latency / scaling study is produced.
    """
    seed = set_seed()
    real = sorted(glob.glob(os.path.join(SOCO_DIR, "Real", "*.BMP")))
    if not real:
        print("SOCOFing 'Real' folder not found under fingerprints/Real. Skipping Exp 5.")
        return
    print(f"\n=== EXPERIMENT 5: SOCOFing SIFT vs Minutiae (real, large 1:N) | seed={seed} ===")
    print(f"Gallery available: {len(real)} real fingerprints (600 subjects x 10 fingers).")
    print(f"Each algorithm uses its OWN best preprocessing (upscale x{SOCO_UPSCALE}).\n")

    paths = real[:sample]
    rows = []
    timings = {}
    print(f"{'algo':10s} {'avg minutiae/kp':>16s} {'extract ms/img':>15s} {'match ms/cmp':>13s}")
    for method in ["SIFT", "Minutiae"]:
        feats, ext_ms = [], []
        for p in paths:
            t0 = perf_counter()
            f = _soco_feature(p, method)
            ext_ms.append((perf_counter() - t0) * 1000)
            feats.append(f)
        mfn = _soco_match(method)
        match_ms = []
        for i in range(len(feats) - 1):
            t0 = perf_counter()
            mfn(feats[i], feats[i + 1])          # different fingers -> realistic impostor cost
            match_ms.append((perf_counter() - t0) * 1000)
        if method == "SIFT":
            cnt = np.mean([f["n_kp"] for f in feats])
        else:
            cnt = np.mean([f["n_minutiae"] for f in feats])
        em, mm = float(np.mean(ext_ms)), float(np.mean(match_ms))
        timings[method] = (em, mm)
        print(f"{method:10s} {cnt:16.0f} {em:15.2f} {mm:13.3f}")
        rows.append({"algorithm": method, "avg_features": round(cnt, 0),
                     "extract_ms_per_img": round(em, 2), "match_ms_per_cmp": round(mm, 3)})

    _write_csv(os.path.join(RESULTS, "exp5_socofing_latency.csv"), rows)

    # Project 1:N identification latency = extract(query) + N * match, for growing N
    Ns = [100, 1000, 6000, 60000]
    print(f"\n--- Projected 1:N identification time per query (extract + N x match) ---")
    print(f"{'N':>8s} " + " ".join(f"{m:>12s}" for m in ['SIFT(s)', 'Minutiae(s)']))
    proj = {m: [] for m in ["SIFT", "Minutiae"]}
    for N in Ns:
        vals = []
        for method in ["SIFT", "Minutiae"]:
            em, mm = timings[method]
            sec = (em + N * mm) / 1000.0
            proj[method].append(sec)
            vals.append(sec)
        print(f"{N:>8d} " + " ".join(f"{v:>12.2f}" for v in vals))

    plt.figure(figsize=(8, 6))
    for method in ["SIFT", "Minutiae"]:
        plt.plot(Ns, proj[method], marker="o", label=method)
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("Gallery size N (log)", fontweight="bold")
    plt.ylabel("Identification time per query (s, log)", fontweight="bold")
    plt.title("Exp 5: 1:N identification cost vs gallery size (SOCOFing)", fontweight="bold")
    plt.legend(); plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig = os.path.join(FIGURES, "exp5_latency_scaling.png")
    plt.savefig(fig, dpi=150); plt.close()
    print(f"\nSaved: results/exp5_socofing_latency.csv\nSaved: {fig}")

    altered = glob.glob(os.path.join(SOCO_DIR, "Altered*", "**", "*.BMP"), recursive=True)
    if not altered:
        print("\n[Accuracy] Skipped: download the SOCOFing 'Altered' folder (genuine pairs)")
        print("           to fingerprints/Altered-* to enable the Rank-1 / identification study.")
    else:
        print(f"\n[Accuracy] Found {len(altered)} altered images — accuracy study will be added next.")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=int, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    runners = {1: experiment_1, 2: experiment_2, 3: experiment_3, 4: experiment_4, 5: experiment_5}
    if args.all:
        for i in [1, 2, 3, 4, 5]:
            runners[i]()
    elif args.exp:
        runners[args.exp]()
    else:
        print("Select: 1=Preprocessing 2=Generalization 3=Algo x DB 4=Scoring 5=SOCOFing(SIFT vs Minutiae)")
        choice = input("Experiment number: ").strip()
        runners.get(int(choice), lambda: print("Invalid"))() if choice.isdigit() else print("Invalid")


if __name__ == "__main__":
    main()
