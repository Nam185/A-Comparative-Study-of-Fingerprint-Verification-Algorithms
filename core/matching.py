"""Matching + scoring. RANSAC uses OpenCV's RNG, so call io_utils.set_seed() first."""
import cv2
import numpy as np

SCORING_VARIANTS = {
    "S1": "absolute good-match count",
    "S2": "% matches (good / min_kp * 100)",
    "S3": "RANSAC inlier count",
    "S4": "RANSAC inlier ratio (inliers / good * 100)",
}


def match_components(featA, featB, ratio=0.75):
    """For keypoint methods: return dict with good, inliers, min_kp (computed once)."""
    out = {"good": 0, "inliers": 0, "min_kp": 0}
    dA, dB = featA.get("desc"), featB.get("desc")
    if dA is None or dB is None or len(dA) < 2 or len(dB) < 2:
        return out
    out["min_kp"] = min(len(dA), len(dB))
    bf = cv2.BFMatcher(featA.get("norm", cv2.NORM_L2))
    matches = bf.knnMatch(dA, dB, k=2)
    good = []
    for pair in matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio * n.distance:
                good.append(m)
    out["good"] = len(good)
    if len(good) < 4:
        return out
    src = np.float32([featA["kp"][m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([featB["kp"][m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    out["inliers"] = 0 if mask is None else int(mask.sum())
    return out


def score_from_components(comp, variant="S3"):
    good, inliers, min_kp = comp["good"], comp["inliers"], comp["min_kp"]
    if variant == "S1":
        return float(good)
    if variant == "S2":
        return (good / min_kp * 100) if min_kp else 0.0
    if variant == "S3":
        return float(inliers)
    if variant == "S4":
        return (inliers / good * 100) if good else 0.0
    raise ValueError(variant)


def chi_square_similarity(vecA, vecB):
    """Block-LBP histograms -> 0..100 similarity (higher = more similar)."""
    chi = 0.5 * np.sum((vecA - vecB) ** 2 / (vecA + vecB + 1e-10))
    return float(np.exp(-chi)) * 100


def match(featA, featB, ratio=0.75, scoring="S3"):
    """Generic match. Returns a similarity score (higher = more similar)."""
    if featA is None or featB is None:
        return 0.0
    if featA["kind"] == "hist":
        return chi_square_similarity(featA["vec"], featB["vec"])
    comp = match_components(featA, featB, ratio)
    return score_from_components(comp, scoring)
