"""I/O helpers, dataset paths and reproducible random seed."""
import os
import cv2
import numpy as np
import random

# Project root = parent of this core/ folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FP_DIR = os.path.join(BASE_DIR, "fingerprints")
DBS = ["DB1_B", "DB2_B", "DB3_B", "DB4_B"]
# Difficulty labels used in the report
DB_DIFFICULTY = {"DB1_B": "average", "DB2_B": "easy", "DB3_B": "hard", "DB4_B": "synthetic/worst"}

SEED = 42  # fixed so RANSAC (OpenCV RNG) gives reproducible EER


def set_seed(seed=SEED):
    """Seed every RNG that can affect results (OpenCV RANSAC uses cv2 RNG)."""
    cv2.setRNGSeed(seed)
    np.random.seed(seed)
    random.seed(seed)
    return seed


def read_gray(path):
    """Read a grayscale image, tolerant of non-ASCII paths."""
    arr = np.fromfile(path, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)


def db_path(db):
    return os.path.join(FP_DIR, db)


def img_path(db, finger_id, impression):
    return os.path.join(FP_DIR, db, f"{finger_id}_{impression}.tif")
