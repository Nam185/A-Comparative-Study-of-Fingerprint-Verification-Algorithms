"""Feature extractors. Each takes an ALREADY-PREPROCESSED grayscale image.

All return a dict so matching.py can stay generic:
  keypoint methods -> {"kind":"kp", "kp":[...], "desc":ndarray, "norm":..., "n_kp":int, ...}
  histogram method -> {"kind":"hist", "vec":ndarray}
"""
import cv2
import numpy as np
from skimage import feature as skfeature
from skimage.morphology import skeletonize

_SIFT = cv2.SIFT_create()
_ORB = cv2.ORB_create(nfeatures=1500)

# ---------------- SIFT ----------------
def sift_features(enh):
    kp, desc = _SIFT.detectAndCompute(enh, None)
    return {"kind": "kp", "kp": kp, "desc": desc, "norm": cv2.NORM_L2,
            "n_kp": 0 if desc is None else len(desc)}

# ---------------- ORB ----------------
def orb_features(enh):
    kp, desc = _ORB.detectAndCompute(enh, None)
    return {"kind": "kp", "kp": kp, "desc": desc, "norm": cv2.NORM_HAMMING,
            "n_kp": 0 if desc is None else len(desc)}

# ---------------- LBP (block-based) ----------------
LBP_RADIUS, LBP_POINTS, LBP_GRID, LBP_SIZE = 2, 16, 8, (256, 256)

def lbp_features(enh):
    img = cv2.resize(enh, LBP_SIZE)
    lbp = skfeature.local_binary_pattern(img, LBP_POINTS, LBP_RADIUS, method="uniform")
    n_bins = LBP_POINTS + 2
    h, w = lbp.shape
    vec = []
    for gy in range(LBP_GRID):
        for gx in range(LBP_GRID):
            cell = lbp[gy*h//LBP_GRID:(gy+1)*h//LBP_GRID, gx*w//LBP_GRID:(gx+1)*w//LBP_GRID]
            hist, _ = np.histogram(cell.ravel(), bins=n_bins, range=(0, n_bins))
            hist = hist.astype("float32"); hist /= (hist.sum() + 1e-7)
            vec.append(hist)
    return {"kind": "hist", "vec": np.concatenate(vec)}

# ---------------- Minutiae (Crossing Number) ----------------
from core.minutiae_native import roi_mask as _roi_mask   # variance+coherence+largest-CC segmentation

def extract_minutiae_points(enh):
    """Return list of (x, y, type) where type is 'end' or 'bif'."""
    bw = cv2.adaptiveThreshold(enh, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, 15, 5)
    skel = skeletonize(bw > 0).astype(np.uint8)
    mask = _roi_mask(enh)
    raw = []
    h, w = skel.shape
    for y in range(1, h - 1):
        row = skel[y]
        for x in range(1, w - 1):
            if row[x] != 1 or mask[y, x] == 0:
                continue
            p = [skel[y-1,x], skel[y-1,x+1], skel[y,x+1], skel[y+1,x+1],
                 skel[y+1,x], skel[y+1,x-1], skel[y,x-1], skel[y-1,x-1]]
            cn = sum(abs(int(p[i]) - int(p[(i+1) % 8])) for i in range(8)) // 2
            if cn == 1:
                raw.append((x, y, 'end'))
            elif cn == 3:
                raw.append((x, y, 'bif'))
    D = 8
    keep = [True] * len(raw)
    for i in range(len(raw)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(raw)):
            if keep[j] and abs(raw[i][0]-raw[j][0]) < D and abs(raw[i][1]-raw[j][1]) < D:
                keep[j] = False
    return [raw[i] for i in range(len(raw)) if keep[i]]

def minutiae_features(enh):
    pts = extract_minutiae_points(enh)
    if len(pts) < 4:
        return {"kind": "kp", "kp": None, "desc": None, "norm": cv2.NORM_L2,
                "n_kp": 0, "n_minutiae": len(pts)}
    kps = [cv2.KeyPoint(float(x), float(y), 12) for (x, y, t) in pts]
    kps, desc = _SIFT.compute(enh, kps)
    return {"kind": "kp", "kp": kps, "desc": desc, "norm": cv2.NORM_L2,
            "n_kp": 0 if desc is None else len(desc), "n_minutiae": len(pts)}

# Registry used by the experiments
EXTRACTORS = {
    "SIFT": sift_features,
    "ORB": orb_features,
    "LBP": lbp_features,
    "Minutiae": minutiae_features,
}
