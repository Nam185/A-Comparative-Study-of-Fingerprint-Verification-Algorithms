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
from core.matching import match, SCORING_VARIANTS
from core.evaluation import collect_scores, eer, FINGERS, IMPRESSIONS

RESULTS = os.path.join(BASE_DIR, "results")
FIGURES = os.path.join(RESULTS, "figures")
os.makedirs(FIGURES, exist_ok=True)

COMBO_ORDER = ["C1", "C2", "C3", "C1+G", "C2+G", "C3+G"]
ALGO_ORDER = ["SIFT", "ORB", "LBP", "Minutiae"]
# Per-algorithm scoring + 1:N fixed threshold (from production defaults)
ALGO_SCORING = {"SIFT": "S3", "ORB": "S3", "LBP": "chi", "Minutiae": "S3"}


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


def experiment_3():
    print("Experiment 3 (full algo x DB, 1:1 + 1:N) — not implemented yet. Coming next.")


def experiment_4():
    print("Experiment 4 (scoring strategy S1-S4) — not implemented yet. Coming next.")


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
