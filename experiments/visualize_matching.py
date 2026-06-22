"""Matching visualization: prove the matcher SEPARATES genuine from impostor.

Draws the geometrically consistent (RANSAC inlier) SIFT matches for:
  - a GENUINE pair (two impressions of the same finger)  -> many consistent lines
  - an IMPOSTOR pair (two different fingers)              -> very few / no lines
Saves results/figures/feature_matching_genuine_vs_impostor.png

Usage:
    python experiments/visualize_matching.py            # default DB2_B genuine vs impostor
    python experiments/visualize_matching.py DB1_B 101  # use DB1_B, anchor finger 101
"""
import os
import sys
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.io_utils import BASE_DIR, read_gray, img_path
from core.preprocessing import c1

FIGDIR = os.path.join(BASE_DIR, "results", "figures")
os.makedirs(FIGDIR, exist_ok=True)
_SIFT = cv2.SIFT_create()


def _feat(path):
    enh = c1(read_gray(path))
    kp, desc = _SIFT.detectAndCompute(enh, None)
    return enh, kp, desc


def match_and_draw(pathA, pathB, max_lines=60):
    enhA, kpA, dA = _feat(pathA)
    enhB, kpB, dB = _feat(pathB)
    bf = cv2.BFMatcher()
    good = [m for m, n in bf.knnMatch(dA, dB, k=2) if m.distance < 0.75 * n.distance]
    inlier_matches, n_inliers = [], 0
    if len(good) >= 4:
        src = np.float32([kpA[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kpB[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if mask is not None:
            inlier_matches = [good[i] for i in range(len(good)) if mask[i]]
            n_inliers = len(inlier_matches)
    visA = cv2.cvtColor(enhA, cv2.COLOR_GRAY2RGB)
    visB = cv2.cvtColor(enhB, cv2.COLOR_GRAY2RGB)
    out = cv2.drawMatches(visA, kpA, visB, kpB, inlier_matches[:max_lines], None,
                          matchColor=(0, 200, 0), flags=2)
    return out, n_inliers


def main():
    db = sys.argv[1] if len(sys.argv) > 1 else "DB2_B"
    anchor = sys.argv[2] if len(sys.argv) > 2 else "101"
    a = int(anchor)
    genuine, g_in = match_and_draw(img_path(db, a, 1), img_path(db, a, 2))
    impostor, i_in = match_and_draw(img_path(db, a, 1), img_path(db, a + 1, 1))

    print(f"DB={db}")
    print(f"  GENUINE  ({a}_1 vs {a}_2):   {g_in} consistent (RANSAC inlier) matches")
    print(f"  IMPOSTOR ({a}_1 vs {a+1}_1): {i_in} consistent matches")

    fig, axes = plt.subplots(2, 1, figsize=(12, 11))
    axes[0].imshow(genuine)
    axes[0].set_title(f"GENUINE pair  ({a}_1 vs {a}_2, same finger) - "
                      f"{g_in} consistent matches", fontsize=13, fontweight="bold", color="green")
    axes[1].imshow(impostor)
    axes[1].set_title(f"IMPOSTOR pair  ({a}_1 vs {a+1}_1, different fingers) - "
                      f"{i_in} consistent matches", fontsize=13, fontweight="bold", color="red")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"SIFT + RANSAC matching - genuine vs impostor ({db})\n"
                 "green lines = geometrically consistent matches",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(FIGDIR, "feature_matching_genuine_vs_impostor.png")
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
