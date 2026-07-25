"""Visualize what the liveness detector (PAD) 'sees'.

Shows the multi-radius LBP texture maps and the 56-D feature vector that
`core.liveness.pad_features` extracts — the exact input to the SVM classifier.
This is the visual proof that PAD feature extraction works (analogous to the
feature-extraction showcase for matching).

Usage:
    python experiments/visualize_liveness.py <live_image> [spoof_image]
    python experiments/visualize_liveness.py DB2_B/101_1.tif          # one image (texture breakdown)
    python experiments/visualize_liveness.py path/live.png path/spoof.png   # live vs spoof comparison
"""
import os
import sys
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage import feature as skf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.io_utils import BASE_DIR, read_gray, img_path
from core.liveness import pad_features, LBP_CONFIG, PAD_SIZE

FIGDIR = os.path.join(BASE_DIR, "results", "figures")
os.makedirs(FIGDIR, exist_ok=True)


def _resolve(arg):
    if os.path.exists(arg):
        return arg
    # try a DB-relative form like "DB2_B/101_1.tif"
    try:
        db, fname = arg.replace("\\", "/").split("/")[-2:]
        fid, imp = os.path.splitext(fname)[0].split("_")
        cand = img_path(db, fid, imp)
        if os.path.exists(cand):
            return cand
    except ValueError:
        pass
    return arg  # hand back as-is; read_gray reports if it cannot be read


def _lbp_map(img, radius, n_points):
    lbp = skf.local_binary_pattern(img, n_points, radius, method="uniform")
    return cv2.normalize(lbp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def _row(fig, gs, r, img, title):
    """Draw one row: image + 3 LBP maps + LBP histogram (the classifier's input)."""
    small = cv2.resize(img, PAD_SIZE)
    small = cv2.normalize(small, None, 0, 255, cv2.NORM_MINMAX)
    panels = [(small, f"{title}")]
    for radius, n_points in LBP_CONFIG:
        panels.append((cv2.applyColorMap(_lbp_map(small, radius, n_points), cv2.COLORMAP_JET)[:, :, ::-1],
                       f"LBP R={radius}"))
    for c, (im, t) in enumerate(panels):
        ax = fig.add_subplot(gs[r, c])
        ax.imshow(im, cmap=None if im.ndim == 3 else "gray")
        ax.set_title(t, fontsize=10, fontweight="bold"); ax.axis("off")
    # feature vector: plot the 54 LBP histogram bins (the 2 quality stats shown as text)
    feat = pad_features(img)
    ax = fig.add_subplot(gs[r, 4])
    ax.bar(range(54), feat[:54], color="steelblue")
    ax.set_title("LBP texture histogram (54 bins)", fontsize=10, fontweight="bold")
    ax.set_xticks([]); ax.set_ylabel("norm. freq.")
    ax.text(0.02, 0.92, f"contrast={feat[54]:.2f}\nsharpness={feat[55]:.2f}",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round", fc="wheat", alpha=0.7))


def main():
    if len(sys.argv) < 2:
        print("Usage: python experiments/visualize_liveness.py <image> [spoof_image]")
        return
    p1 = _resolve(sys.argv[1])
    img1 = read_gray(p1)
    if img1 is None:
        print(f"Cannot read {p1}"); return

    if len(sys.argv) >= 3:  # live vs spoof comparison
        p2 = _resolve(sys.argv[2]); img2 = read_gray(p2)
        fig = plt.figure(figsize=(18, 7))
        gs = fig.add_gridspec(2, 5)
        _row(fig, gs, 0, img1, "LIVE")
        _row(fig, gs, 1, img2, "SPOOF")
        fig.suptitle("Liveness features: LIVE vs SPOOF (texture + 56-D vector)",
                     fontsize=15, fontweight="bold")
        out = os.path.join(FIGDIR, "liveness_features_live_vs_spoof.png")
    else:  # single image texture breakdown
        fig = plt.figure(figsize=(18, 4))
        gs = fig.add_gridspec(1, 5)
        _row(fig, gs, 0, img1, os.path.basename(p1))
        fig.suptitle("What the liveness detector extracts (multi-radius LBP texture + feature vector)",
                     fontsize=14, fontweight="bold")
        out = os.path.join(FIGDIR, "liveness_features_single.png")

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"Feature vector length: {len(pad_features(img1))}")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
