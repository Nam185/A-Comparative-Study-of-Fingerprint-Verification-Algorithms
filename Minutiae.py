"""
Minutiae & Bifurcation method (classical fingerprint approach).

Pipeline: preprocess -> binarize -> thinning (skeleton) -> Crossing Number ->
remove spurious minutiae. A minutia is a ridge ENDING (CN=1) or a BIFURCATION (CN=3).

For matching we treat the detected minutiae as keypoints, describe their local
neighbourhood with a SIFT descriptor, then match + RANSAC (same geometric
verification as the other methods). This keeps the comparison fair.

NOTE (honest finding): on FVC2002 this matcher reaches ~18-20% EER on DB1_B, i.e.
comparable to but NOT better than full SIFT. The reason is that minutiae extraction
is very sensitive to binarization quality: broken ridges create many spurious ridge
endings. The real value of this module is (a) the explainable classical algorithm
and (b) the minutiae visualization (menu option 4), which is great for the demo.
"""
import cv2
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_FOLDER = os.path.join(BASE_DIR, "fingerprints", "DB1_B")

_SIFT = cv2.SIFT_create()


def _preprocess(img):
    """Normalize + Gaussian Blur + CLAHE (same Combo 1 as the other methods)."""
    n = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    b = cv2.GaussianBlur(n, (3, 3), 0)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(b)


def _roi_mask(enh, block=16, var_thresh=100):
    """Segment the fingerprint region by local variance (ridges = high variance)."""
    h, w = enh.shape
    mask = np.zeros((h, w), np.uint8)
    f = enh.astype(np.float32)
    for y in range(0, h, block):
        for x in range(0, w, block):
            if f[y:y + block, x:x + block].var() > var_thresh:
                mask[y:y + block, x:x + block] = 1
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    # Erode the border so unreliable minutiae near the edge are dropped
    mask = cv2.erode(mask, np.ones((10, 10), np.uint8))
    return mask


def extract_minutiae(img):
    """
    Extract minutiae with the Crossing Number algorithm.
    Returns (minutiae_list, skeleton) where each minutia is (x, y, type),
    type is 'end' (ridge ending) or 'bif' (bifurcation).
    """
    enh = _preprocess(img)
    # Adaptive threshold; INV so ridges become white (value 1) after skeletonize
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
            # 8 neighbours clockwise
            p = [skel[y - 1, x], skel[y - 1, x + 1], skel[y, x + 1], skel[y + 1, x + 1],
                 skel[y + 1, x], skel[y + 1, x - 1], skel[y, x - 1], skel[y - 1, x - 1]]
            # Crossing Number = half the sum of absolute neighbour differences
            cn = sum(abs(int(p[i]) - int(p[(i + 1) % 8])) for i in range(8)) // 2
            if cn == 1:
                raw.append((x, y, 'end'))    # ridge ending
            elif cn == 3:
                raw.append((x, y, 'bif'))    # bifurcation

    # Remove spurious minutiae: drop one of any pair closer than D px (broken ridges)
    D = 8
    keep = [True] * len(raw)
    for i in range(len(raw)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(raw)):
            if keep[j] and abs(raw[i][0] - raw[j][0]) < D and abs(raw[i][1] - raw[j][1]) < D:
                keep[j] = False
    minutiae = [raw[i] for i in range(len(raw)) if keep[i]]
    return minutiae, skel


def process_single_image(image_path):
    """
    Extract minutiae, then describe each minutia with a local SIFT descriptor.
    Returns a (keypoints, descriptors) tuple, like the SIFT/ORB modules.
    """
    img_array = np.fromfile(image_path, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: cannot read image at {image_path}")
        return None

    pts, _ = extract_minutiae(img)
    if len(pts) < 4:
        return (None, None)

    enh = _preprocess(img)
    kps = [cv2.KeyPoint(float(x), float(y), 12) for (x, y, t) in pts]
    kps, desc = _SIFT.compute(enh, kps)
    return (kps, desc)


def match_two_fingerprints(feat_A, feat_B, ratio=0.8):
    """Match minutiae descriptors with ratio test + RANSAC. Score = inlier count."""
    if feat_A is None or feat_B is None:
        return 0.0
    kp_A, desc_A = feat_A
    kp_B, desc_B = feat_B
    if desc_A is None or desc_B is None or len(desc_A) < 2 or len(desc_B) < 2:
        return 0.0

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc_A, desc_B, k=2)
    good = []
    for pair in matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio * n.distance:
                good.append(m)
    if len(good) < 4:
        return float(len(good))

    src = np.float32([kp_A[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp_B[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if mask is None:
        return 0.0
    return float(int(mask.sum()))


def num_keypoints(feat):
    if feat is None or feat[1] is None:
        return 0
    return len(feat[1])


def build_database(folder_path):
    """Step 1 (1:N): Enrollment. Returns {person_id: (keypoints, descriptors)}."""
    database = {}
    print("Loading fingerprints into the system (Database)...")
    for student_id in range(101, 111):
        img_path = os.path.join(folder_path, f"{student_id}_1.tif")
        feat = process_single_image(img_path)
        if feat is not None and feat[1] is not None:
            database[str(student_id)] = feat
    print(f"-> Done! Stored templates of {len(database)} students.")
    return database


def identify_fingerprint(query_feat, database, threshold=10):
    """Step 2 (1:N): Identification. Score = number of matched minutiae (inliers)."""
    if query_feat is None or query_feat[1] is None:
        return "Image error", 0
    best_match_id, highest_score = "Unknown", 0
    for student_id, db_feat in database.items():
        score = match_two_fingerprints(query_feat, db_feat)
        if score > highest_score:
            highest_score, best_match_id = score, student_id
    if highest_score < threshold:
        return "Imposter", highest_score
    return best_match_id, highest_score


def calculate_metrics(genuine_scores, imposter_scores, thresholds):
    """Compute FAR, FRR arrays and find the EER."""
    far_list, frr_list = [], []
    for thresh in thresholds:
        far = sum(1 for s in imposter_scores if s >= thresh) / len(imposter_scores) if imposter_scores else 0
        frr = sum(1 for s in genuine_scores if s < thresh) / len(genuine_scores) if genuine_scores else 0
        far_list.append(far)
        frr_list.append(frr)
    differences = [abs(far - frr) for far, frr in zip(far_list, frr_list)]
    min_idx = int(np.argmin(differences))
    eer = (far_list[min_idx] + frr_list[min_idx]) / 2
    return eer, thresholds[min_idx]


def show_minutiae(image_path):
    """Visualize minutiae: red circles = ridge endings, blue = bifurcations."""
    img_array = np.fromfile(image_path, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: cannot read image at {image_path}")
        return
    pts, _ = extract_minutiae(img)
    vis = cv2.cvtColor(_preprocess(img), cv2.COLOR_GRAY2RGB)
    n_end = n_bif = 0
    for (x, y, t) in pts:
        if t == 'end':
            cv2.circle(vis, (x, y), 5, (255, 0, 0), 1)   # red = ending
            n_end += 1
        else:
            cv2.circle(vis, (x, y), 5, (0, 128, 255), 2)  # blue = bifurcation
            n_bif += 1
    print(f"Detected {len(pts)} minutiae: {n_end} ridge endings, {n_bif} bifurcations")
    plt.figure(figsize=(7, 8))
    plt.imshow(vis)
    plt.title(f"Minutiae map - {os.path.basename(image_path)}\n"
              f"red = ridge ending ({n_end}), blue = bifurcation ({n_bif})")
    plt.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    folder_path = DEFAULT_FOLDER

    while True:
        print("\n" + "=" * 50)
        print("=== FINGERPRINT VERIFICATION SYSTEM - MINUTIAE (Crossing Number) ===")
        print("=" * 50)
        print("1. Run 1:1 performance evaluation (FAR, FRR, EER)")
        print("2. Start 1:N class attendance system")
        print("3. Visualize minutiae of one image (demo)")
        print("4. Exit")

        choice = input("\nSelect a feature (1/2/3/4): ")

        if choice == '1':
            genuine_scores, imposter_scores, execution_times = [], [], []
            print("\n[Running] Scanning data and computing. Please wait...\n")

            cache = {}
            def feat(fid, imp):
                key = (fid, imp)
                if key not in cache:
                    cache[key] = process_single_image(os.path.join(folder_path, f"{fid}_{imp}.tif"))
                return cache[key]

            for finger_id in range(101, 111):
                base = feat(finger_id, 1)
                for impression in range(2, 9):
                    start_time = time.time()
                    score = match_two_fingerprints(base, feat(finger_id, impression))
                    execution_times.append(time.time() - start_time)
                    genuine_scores.append(score)

            for i in range(101, 111):
                a = feat(i, 1)
                for j in range(i + 1, 111):
                    imposter_scores.append(match_two_fingerprints(a, feat(j, 1)))

            avg_latency = np.mean(execution_times)
            eer, optimal_threshold = calculate_metrics(genuine_scores, imposter_scores, range(0, 100))

            print("=== PERFORMANCE REPORT (1:1 VERIFICATION) ===")
            print(f"Genuine trials: {len(genuine_scores)}")
            print(f"Imposter trials: {len(imposter_scores)}")
            print(f"Genuine score (avg): {np.mean(genuine_scores):.1f}  |  Imposter score (avg): {np.mean(imposter_scores):.1f}")
            print(f"1. Average latency: {avg_latency:.4f} s/sample")
            print(f"2. Optimal threshold: {optimal_threshold} matched minutiae")
            print(f"3. Equal Error Rate (EER): {eer * 100:.2f}%")

            input("\nPress Enter to return to the main menu...")

        elif choice == '2':
            print("\n[System] Starting the scanner and loading the database...")
            class_database = build_database(folder_path)
            attendance_threshold = 10

            while True:
                print(f"\n--- ATTENDANCE SCANNER READY (safety threshold: {attendance_threshold} minutiae) ---")
                test_file = input("Enter fingerprint file name (e.g. 105_2.tif) or 'exit': ")
                if test_file.lower() == 'exit':
                    break
                img_test_path = os.path.join(folder_path, test_file)
                if not os.path.exists(img_test_path):
                    print("Error: file not found! Please try again.")
                    continue
                print("Analyzing fingerprint...")
                desc_query = process_single_image(img_test_path)
                if num_keypoints(desc_query) == 0:
                    print("\n" + "-" * 30)
                    print("REJECTED: no minutiae found (image too blurry or corrupted)!")
                    print("-" * 30)
                    continue
                predicted_id, highest_score = identify_fingerprint(desc_query, class_database, threshold=attendance_threshold)
                print("\n" + "-" * 30)
                if predicted_id == "Imposter":
                    print(f"WARNING: IMPOSTER DETECTED! (best match score only {highest_score:.0f})")
                else:
                    print(f"ATTENDANCE OK: Student ID {predicted_id} (matched minutiae: {highest_score:.0f})")
                print("-" * 30)

        elif choice == '3':
            test_file = input("Enter fingerprint file name (e.g. 101_1.tif) or full path: ").strip().strip('"').strip("'")
            p = test_file if os.path.exists(test_file) else os.path.join(folder_path, test_file)
            if not os.path.exists(p):
                print("Error: file not found!")
            else:
                show_minutiae(p)
            input("\nPress Enter to return to the menu...")
        elif choice == '4':
            print("\nThank you for using the system. Goodbye!")
            break
        else:
            print("Invalid choice, please enter 1, 2, 3 or 4.")
