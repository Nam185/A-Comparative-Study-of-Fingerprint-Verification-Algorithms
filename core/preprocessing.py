"""Preprocessing combos for the comparison study.

Three base combos (denoise + contrast) and their Gabor-enhanced variants:
    C1   = Gaussian Blur + CLAHE
    C2   = Bilateral Filter + CLAHE
    C3   = Median Blur + Histogram Equalization
    C1+G = C1 followed by a Gabor filter bank   (likewise C2+G, C3+G)

The production demo apps keep using C1. These combos exist for the experiments.
"""
import cv2
import numpy as np

_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def _normalize(img):
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)


def c1(img):
    """Gaussian Blur + CLAHE (production default)."""
    n = _normalize(img)
    b = cv2.GaussianBlur(n, (3, 3), 0)
    return _CLAHE.apply(b)


def c2(img):
    """Bilateral Filter (edge-preserving denoise) + CLAHE."""
    n = _normalize(img)
    b = cv2.bilateralFilter(n, 5, 50, 50)
    return _CLAHE.apply(b)


def c3(img):
    """Median Blur + global Histogram Equalization."""
    n = _normalize(img)
    b = cv2.medianBlur(n, 3)
    return cv2.equalizeHist(b)


# --- Gabor filter bank: orientation-independent ridge enhancement ---
_GABOR_KERNELS = [
    cv2.getGaborKernel((21, 21), sigma=4.0, theta=theta, lambd=10.0, gamma=0.5, psi=0)
    for theta in np.arange(0, np.pi, np.pi / 8)
]


def gabor_enhance(img):
    """Apply a bank of oriented Gabor filters and keep the max response per pixel."""
    acc = np.zeros_like(img, dtype=np.float32)
    src = img.astype(np.float32)
    for k in _GABOR_KERNELS:
        acc = np.maximum(acc, cv2.filter2D(src, cv2.CV_32F, k))
    return cv2.normalize(acc, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def _with_gabor(base):
    def fn(img):
        return gabor_enhance(base(img))
    return fn


# Ordered dict of all 6 combos used by the experiments
COMBOS = {
    "C1": c1,
    "C2": c2,
    "C3": c3,
    "C1+G": _with_gabor(c1),
    "C2+G": _with_gabor(c2),
    "C3+G": _with_gabor(c3),
}

COMBO_DESC = {
    "C1": "Gaussian + CLAHE",
    "C2": "Bilateral + CLAHE",
    "C3": "Median + HistEq",
    "C1+G": "Gaussian + CLAHE + Gabor",
    "C2+G": "Bilateral + CLAHE + Gabor",
    "C3+G": "Median + HistEq + Gabor",
}
