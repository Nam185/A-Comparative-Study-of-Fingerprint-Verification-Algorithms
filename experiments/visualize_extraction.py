"""Feature-extraction showcase: prove each algorithm extracts sensible features.

Runs the ACTUAL extractors from core/ on one fingerprint and renders, side by side:
  original -> preprocessing -> SIFT keypoints / ORB keypoints / LBP texture map /
  Minutiae map (ridge endings + bifurcations + orientation).
Saves results/figures/feature_extraction_showcase.png (+ a standalone minutiae image).

Usage:
    python experiments/visualize_extraction.py [path-or-DBname/file]
    python experiments/visualize_extraction.py            # default DB1_B/101_1.tif
"""
import os
import sys
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage import feature as skfeature

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.io_utils import BASE_DIR, read_gray, img_path
from core.preprocessing import c1
from core.features import sift_features, orb_features, LBP_RADIUS, LBP_POINTS
from core import minutiae_native as MN

FIGDIR = os.path.join(BASE_DIR, "results", "figures")
os.makedirs(FIGDIR, exist_ok=True)


def draw_keypoints(enh, kp, rich=True):
    vis = cv2.cvtColor(enh, cv2.COLOR_GRAY2RGB)
    flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS if rich else 0
    return cv2.drawKeypoints(vis, kp, None, color=(255, 60, 0), flags=flags)


def lbp_map(enh):
    lbp = skfeature.local_binary_pattern(enh, LBP_POINTS, LBP_RADIUS, method="uniform")
    return cv2.normalize(lbp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def draw_minutiae(enh, minu, line_len=10):
    vis = cv2.cvtColor(enh, cv2.COLOR_GRAY2RGB)
    n_end = n_bif = 0
    for (x, y, theta, t) in minu:
        x, y = int(x), int(y)
        if int(t) == 0:
            cv2.circle(vis, (x, y), 5, (255, 0, 0), 1)      # ending = red
            n_end += 1
        else:
            cv2.circle(vis, (x, y), 5, (0, 128, 255), 2)    # bifurcation = blue
            n_bif += 1
        # orientation tick
        x2 = int(x + line_len * np.cos(theta)); y2 = int(y + line_len * np.sin(theta))
        cv2.line(vis, (x, y), (x2, y2), (0, 200, 0), 1)
    return vis, n_end, n_bif


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "DB1_B/101_1.tif"
    if os.path.exists(arg):
        path = arg
    elif "/" in arg or "\\" in arg:
        db, fname = arg.replace("\\", "/").split("/")
        fid, imp = fname.replace(".tif", "").split("_")
        path = img_path(db, fid, imp)
    else:
        path = arg
    img = read_gray(path)
    if img is None:
        print(f"Cannot read {path}"); return
    enh = c1(img)

    f_sift = sift_features(enh)
    f_orb = orb_features(enh)
    minu = MN.extract_minutiae(enh)
    sift_vis = draw_keypoints(enh, f_sift["kp"], rich=True)    # rich = show scale + orientation
    orb_vis = draw_keypoints(enh, f_orb["kp"], rich=False)     # plain dots (1500 pts) for clarity
    lbp_vis = lbp_map(enh)
    min_vis, n_end, n_bif = draw_minutiae(enh, minu)

    print(f"Sample: {os.path.basename(path)}")
    print(f"  SIFT keypoints : {f_sift['n_kp']}")
    print(f"  ORB keypoints  : {f_orb['n_kp']}")
    print(f"  Minutiae       : {len(minu)} ({n_end} endings, {n_bif} bifurcations)")

    panels = [
        (cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), "1. Original grayscale"),
        (cv2.cvtColor(enh, cv2.COLOR_GRAY2RGB), "2. Preprocessed (Normalize+Blur+CLAHE)"),
        (sift_vis, f"3. SIFT keypoints ({f_sift['n_kp']})"),
        (orb_vis, f"4. ORB keypoints ({f_orb['n_kp']})"),
        (cv2.applyColorMap(lbp_vis, cv2.COLORMAP_JET)[:, :, ::-1], "5. LBP texture map (uniform)"),
        (min_vis, f"6. Minutiae: {n_end} endings (red) + {n_bif} bif (blue), green=orientation"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for ax, (im, title) in zip(axes.ravel(), panels):
        ax.imshow(im); ax.set_title(title, fontsize=11, fontweight="bold"); ax.axis("off")
    fig.suptitle(f"Feature Extraction Showcase - {os.path.basename(path)}",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(FIGDIR, "feature_extraction_showcase.png")
    fig.savefig(out, dpi=150); plt.close(fig)

    # standalone, larger minutiae image (the most important for the traditional method)
    plt.figure(figsize=(7, 8)); plt.imshow(min_vis)
    plt.title(f"Minutiae extraction (Crossing Number)\n{n_end} endings (red), {n_bif} bifurcations (blue), orientation (green)",
              fontsize=11, fontweight="bold")
    plt.axis("off"); plt.tight_layout()
    out2 = os.path.join(FIGDIR, "feature_extraction_minutiae.png")
    plt.savefig(out2, dpi=150); plt.close()

    print(f"\nSaved: {out}\nSaved: {out2}")


if __name__ == "__main__":
    main()
