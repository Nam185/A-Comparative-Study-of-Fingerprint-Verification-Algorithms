"""Dedicated minutiae matcher (NO SIFT descriptors).

Each minutia is (x, y, theta, type):
  - (x, y)  : location from the Crossing Number on a thinned ridge map
  - theta   : local ridge orientation (gradient-based orientation field)
  - type    : 0 = ridge ending, 1 = bifurcation

Matching uses ONLY minutiae geometry:
  1. Describe each minutia by its spatial relationship to its k nearest minutiae
     (distances + angles + orientation differences) -> translation/rotation invariant.
  2. Match those descriptors between two prints (L2 + ratio test).
  3. Geometrically verify the matches with a rigid (rotation+translation+scale)
     transform via RANSAC; the score is the number of consistent minutiae.

This is the classical "minutiae alignment" idea, kept simple and explainable, and is
fast because a print has only ~50-150 minutiae (vs ~2000 SIFT keypoints).
"""
import cv2
import numpy as np
from skimage.morphology import skeletonize

K_NEIGHBORS = 5   # neighbours used to describe each minutia


def orientation_field(gray, block=12):
    """Gradient-based ridge orientation (radians in [0, pi)) sampled per pixel-block."""
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gxx = cv2.boxFilter(gx * gx, -1, (block, block))
    gyy = cv2.boxFilter(gy * gy, -1, (block, block))
    gxy = cv2.boxFilter(gx * gy, -1, (block, block))
    # Dominant orientation of the local gradient structure tensor
    theta = 0.5 * np.arctan2(2 * gxy, gxx - gyy) + np.pi / 2
    return np.mod(theta, np.pi)


def roi_mask(enh, block=16, var_thresh=100, coh_thresh=0.35):
    """Segment the fingerprint from the background.

    Variance alone fails when the background is textured/striped (e.g. the DB3
    capacitive sensor), because noisy background also has high variance. We therefore
    require BOTH high local variance AND high gradient-orientation **coherence** (real
    ridges flow consistently; noise does not), then keep only the largest connected
    component so detached background patches are dropped.
    """
    h, w = enh.shape
    gx = cv2.Sobel(enh, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(enh, cv2.CV_64F, 0, 1, ksize=3)
    Gxx = cv2.boxFilter(gx * gx, -1, (block, block))
    Gyy = cv2.boxFilter(gy * gy, -1, (block, block))
    Gxy = cv2.boxFilter(gx * gy, -1, (block, block))
    coherence = np.sqrt((Gxx - Gyy) ** 2 + 4 * Gxy ** 2) / (Gxx + Gyy + 1e-6)

    var = np.zeros((h, w), np.float32)
    f = enh.astype(np.float32)
    for y in range(0, h, block):
        for x in range(0, w, block):
            var[y:y + block, x:x + block] = f[y:y + block, x:x + block].var()

    mask = ((var > var_thresh) & (coherence > coh_thresh)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((11, 11), np.uint8))
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if num > 1:
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = (labels == biggest).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    return cv2.erode(mask, np.ones((10, 10), np.uint8))


# backward-compatible alias
_roi_mask = roi_mask


def extract_minutiae(enh):
    """Return an (N, 4) array of (x, y, theta, type) minutiae from a preprocessed image."""
    bw = cv2.adaptiveThreshold(enh, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, 15, 5)
    skel = skeletonize(bw > 0).astype(np.uint8)
    mask = _roi_mask(enh)
    ofield = orientation_field(enh)

    raw = []
    h, w = skel.shape
    for y in range(1, h - 1):
        row = skel[y]
        for x in range(1, w - 1):
            if row[x] != 1 or mask[y, x] == 0:
                continue
            p = [skel[y-1, x], skel[y-1, x+1], skel[y, x+1], skel[y+1, x+1],
                 skel[y+1, x], skel[y+1, x-1], skel[y, x-1], skel[y-1, x-1]]
            cn = sum(abs(int(p[i]) - int(p[(i+1) % 8])) for i in range(8)) // 2
            if cn == 1:
                raw.append((x, y, float(ofield[y, x]), 0))   # ending
            elif cn == 3:
                raw.append((x, y, float(ofield[y, x]), 1))   # bifurcation

    # Drop spurious minutiae: one of any pair closer than D px (broken-ridge artifacts)
    D = 8
    keep = [True] * len(raw)
    for i in range(len(raw)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(raw)):
            if keep[j] and abs(raw[i][0]-raw[j][0]) < D and abs(raw[i][1]-raw[j][1]) < D:
                keep[j] = False
    return np.array([raw[i] for i in range(len(raw)) if keep[i]], dtype=np.float32).reshape(-1, 4)


def _descriptors(minu, k=K_NEIGHBORS):
    """Invariant local descriptor per minutia from its k nearest neighbours.

    For each neighbour: [distance, angle-to-neighbour relative to own orientation,
    neighbour-orientation relative to own]. Sorted by distance -> order invariant.
    Returns an (N, 3k) float32 array aligned with `minu`.
    """
    n = len(minu)
    desc = np.zeros((n, 3 * k), np.float32)
    if n <= k:
        return desc
    xy = minu[:, :2]
    for i in range(n):
        d = np.linalg.norm(xy - xy[i], axis=1)
        order = np.argsort(d)[1:k + 1]          # k nearest (skip self)
        vec = []
        for j in order:
            dx, dy = xy[j] - xy[i]
            ang = np.mod(np.arctan2(dy, dx) - minu[i, 2], 2 * np.pi)
            dori = np.mod(minu[j, 2] - minu[i, 2], np.pi)
            vec += [d[j], ang, dori]
        desc[i] = vec
    # Scale distance columns down so they don't dominate the L2 distance
    desc[:, 0::3] *= 0.05
    return desc


def minutiae_features(enh):
    """Feature object for the matcher: dict with minutiae array + their descriptors."""
    minu = extract_minutiae(enh)
    desc = _descriptors(minu)
    return {"kind": "minutiae", "minu": minu, "desc": desc, "n_minutiae": len(minu)}


def match(featA, featB, ratio=0.8):
    """Score = number of geometrically consistent matched minutiae (RANSAC inliers)."""
    if featA is None or featB is None:
        return 0.0
    mA, dA = featA["minu"], featA["desc"]
    mB, dB = featB["minu"], featB["desc"]
    if len(mA) < 4 or len(mB) < 4:
        return 0.0

    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn = bf.knnMatch(dA, dB, k=2)
    good = []
    for pair in knn:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio * n.distance:
                good.append(m)
    if len(good) < 4:
        return float(len(good))

    src = np.float32([mA[m.queryIdx, :2] for m in good]).reshape(-1, 1, 2)
    dst = np.float32([mB[m.trainIdx, :2] for m in good]).reshape(-1, 1, 2)
    # Rigid-ish transform (rotation + translation + scale) is right for fingerprints
    _, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                             ransacReprojThreshold=8.0)
    if inliers is None:
        return 0.0
    return float(int(inliers.sum()))
