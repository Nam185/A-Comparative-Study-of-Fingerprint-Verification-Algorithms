"""Fingerprint Presentation Attack Detection (PAD) — liveness detection.

Distinguishes a LIVE finger from a SPOOF (gelatin/latex/silicone/wood-glue/Play-Doh
replica, etc.). Unlike the matching study (which is zero-training), PAD is a supervised
binary classification problem: we must TRAIN a classifier on labelled live/spoof data.

Classical, explainable pipeline (the standard LivDet baseline):
    grayscale -> resize -> multi-radius uniform LBP texture histograms (+ 2 quality
    statistics) -> feature vector -> SVM classifier.

The idea: a real finger and a replica differ in fine surface texture, ridge sharpness
and perspiration pattern, which the local texture descriptor captures.
"""
import os
import cv2
import numpy as np
from skimage import feature as skf

# Multi-radius uniform LBP captures texture at several scales (classic PAD feature)
LBP_CONFIG = [(1, 8), (2, 16), (3, 24)]
PAD_SIZE = (150, 150)

LIVE_KEYS = ("live", "real", "bonafide", "genuine", "alive")
SPOOF_KEYS = ("fake", "spoof", "attack", "false", "gummy", "artificial")

IMG_EXT = (".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff", ".wsq")


def pad_features(img):
    """Return a fixed-length PAD feature vector from a grayscale image."""
    if img is None:
        return None
    img = cv2.resize(img, PAD_SIZE)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    feats = []
    for radius, n_points in LBP_CONFIG:
        lbp = skf.local_binary_pattern(img, n_points, radius, method="uniform")
        n_bins = n_points + 2
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
        hist = hist.astype(np.float32)
        hist /= (hist.sum() + 1e-7)
        feats.append(hist)
    # Two cheap quality statistics: contrast and sharpness (spoofs are often smoother)
    sharpness = cv2.Laplacian(img, cv2.CV_64F).var()
    feats.append(np.array([img.std() / 128.0, sharpness / 1000.0], np.float32))
    return np.concatenate(feats).astype(np.float32)


def _label_for(path):
    """Infer 1=live / 0=spoof from a file's folder names (spoof checked first)."""
    parts = [p.lower() for p in path.replace("\\", "/").split("/")]
    for part in reversed(parts):
        if any(k in part for k in SPOOF_KEYS):
            return 0
        if any(k in part for k in LIVE_KEYS):
            return 1
    return None


def load_live_spoof(root):
    """Walk `root`, returning [(path, label)] with label 1=live, 0=spoof.

    Auto-detects class from folder names, so it works with LivDet-style layouts
    (.../Live/..., .../Fake/Gelatine/...) or a simple Live/ + Spoof/ split.
    """
    items = []
    for dirpath, _, files in os.walk(root):
        label = _label_for(dirpath)
        if label is None:
            continue
        for f in files:
            if f.lower().endswith(IMG_EXT):
                items.append((os.path.join(dirpath, f), label))
    return items


def pad_metrics(y_true, y_pred):
    """Standard PAD metrics (ISO/IEC 30107-3). live=1, spoof=0.

    APCER = spoofs accepted as live; BPCER = live rejected as spoof; ACER = mean.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    spoof = y_true == 0
    live = y_true == 1
    apcer = float(np.mean(y_pred[spoof] == 1)) * 100 if spoof.any() else 0.0
    bpcer = float(np.mean(y_pred[live] == 0)) * 100 if live.any() else 0.0
    acer = (apcer + bpcer) / 2
    acc = float(np.mean(y_pred == y_true)) * 100
    return {"accuracy": acc, "APCER": apcer, "BPCER": bpcer, "ACER": acer}
