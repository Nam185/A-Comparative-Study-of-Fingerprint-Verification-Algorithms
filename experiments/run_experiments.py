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
import argparse
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


# ============================== EXPERIMENT 3 ==============================
def experiment_3():
    seed = set_seed()
    print(f"\n=== EXPERIMENT 3: Full comparison (combo C1, 4 algos x 4 DBs, 1:1 + 1:N) | seed={seed} ===\n")
    rows_11, rows_1n = [], []
    eer_matrix = {db: {} for db in DBS}
    curves = {}  # DB1_B genuine/imposter scores for the ROC figure

    print(f"{'DB':7s} {'algo':10s} | {'1:1 EER%':>8s} {'genAvg':>8s} {'impAvg':>7s} | "
          f"{'thr':>4s} {'Rank1%':>7s} {'IDrate%':>8s}")
    for db in DBS:
        for algo in ALGO_ORDER:
            feat, _ = _build_feat_fn(db, COMBOS["C1"], EXTRACTORS[algo])
            match_fn = lambda a, b: match(a, b, scoring="S3")  # S3 for kp; LBP dispatches to chi
            # --- 1:1 ---
            gen, imp = collect_scores(feat, match_fn)
            e, _ = eer(gen, imp)
            eer_matrix[db][algo] = e
            if db == "DB1_B":
                curves[algo] = (gen, imp)
            # --- 1:N ---
            thr = ALGO_THRESHOLD[algo]
            rank1, idrate, total = rank1_and_idrate(feat, match_fn, thr)
            print(f"{db:7s} {algo:10s} | {e:8.2f} {np.mean(gen):8.1f} {np.mean(imp):7.2f} | "
                  f"{thr:4d} {rank1:7.1f} {idrate:8.1f}")
            rows_11.append({"db": db, "difficulty": DB_DIFFICULTY[db], "algo": algo,
                            "eer_pct": round(e, 2), "genuine_avg": round(float(np.mean(gen)), 2),
                            "imposter_avg": round(float(np.mean(imp)), 2)})
            rows_1n.append({"db": db, "difficulty": DB_DIFFICULTY[db], "algo": algo,
                            "threshold": thr, "rank1_acc_pct": round(rank1, 1),
                            "identification_rate_pct": round(idrate, 1), "n_queries": total})

    _write_csv(os.path.join(RESULTS, "exp3_algo_x_db_1to1.csv"), rows_11)
    _write_csv(os.path.join(RESULTS, "exp3_algo_x_db_1toN.csv"), rows_1n)
    fig1 = os.path.join(FIGURES, "exp3_eer_algo_x_db.png")
    _grouped_bar(eer_matrix, DBS, ALGO_ORDER, "EER (%)",
                 "Exp 3: 1:1 EER per algorithm across databases (combo C1)", fig1, group_by="db")
    fig2 = os.path.join(FIGURES, "exp3_roc_db1.png")
    _roc_curve(curves, fig2, "Exp 3: ROC curve of the 4 algorithms (DB1_B)")
    print(f"\nSaved: exp3_algo_x_db_1to1.csv, exp3_algo_x_db_1toN.csv\nSaved: {fig1}\nSaved: {fig2}")
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=int, choices=[1, 2, 3, 4])
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    runners = {1: experiment_1, 2: experiment_2, 3: experiment_3, 4: experiment_4}
    if args.all:
        for i in [1, 2, 3, 4]:
            runners[i]()
    elif args.exp:
        runners[args.exp]()
    else:
        print("Select experiment: 1=Preprocessing(SIFT)  2=Generalization  3=Algo x DB  4=Scoring")
        choice = input("Experiment number: ").strip()
        runners.get(int(choice), lambda: print("Invalid"))() if choice.isdigit() else print("Invalid")


if __name__ == "__main__":
    main()
