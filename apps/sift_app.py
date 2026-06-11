import cv2
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import glob

# Project root (parent of apps/) -> dataset paths work regardless of working dir
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_FOLDER = os.path.join(BASE_DIR, "fingerprints", "DB1_B")

# Create the SIFT detector only once (re-creating it per image is very slow)
_SIFT = cv2.SIFT_create()


def match_two_fingerprints(feat_A, feat_B, ratio=0.75):
    """
    Match two fingerprints with SIFT + Lowe's Ratio Test + GEOMETRIC VERIFICATION (RANSAC).

    feat_A, feat_B are (keypoints, descriptors) tuples returned by process_single_image.
    The returned score = number of GEOMETRICALLY CONSISTENT matches (RANSAC inliers).

    Why RANSAC? Two different fingerprints can still produce a few accidental
    descriptor matches (noise). Those matches are scattered and do NOT follow a single
    consistent geometric transform. RANSAC keeps only the matches whose point
    positions transform consistently -> impostor scores drop close to 0, while
    genuine scores stay high. This is the single biggest accuracy improvement.
    """
    if feat_A is None or feat_B is None:
        return 0.0

    kp_A, desc_A = feat_A
    kp_B, desc_B = feat_B
    if desc_A is None or desc_B is None or len(desc_A) < 2 or len(desc_B) < 2:
        return 0.0

    # Step 1: brute-force matching + Lowe's Ratio Test to filter ambiguous matches
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc_A, desc_B, k=2)
    good_matches = []
    for pair in matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio * n.distance:
                good_matches.append(m)

    # Need at least 4 correspondences to estimate a homography
    if len(good_matches) < 4:
        return float(len(good_matches))

    # Step 2: geometric verification with RANSAC
    src_pts = np.float32([kp_A[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_B[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if mask is None:
        return 0.0

    # Inlier count = number of truly geometrically consistent matches
    inliers = int(mask.sum())
    return float(inliers)


def build_mega_database(folder_paths):
    """
    Step 1 (1:N): Enrollment from MULTIPLE folders.
    `folder_paths` is a list of folder paths to merge, e.g. ['.../DB1_B', '.../DB2_B'].
    """
    database = {}
    print(f"\n[System] Loading fingerprints from {len(folder_paths)} different sources...")

    for folder in folder_paths:
        db_name = folder.replace('\\', '/').split('/')[-1]
        file_pattern = os.path.join(folder, "*_1.tif")
        image_files = glob.glob(file_pattern)

        if not image_files:
            print(f"Warning: no template image (_1.tif) found in {folder}")
            continue

        for img_path in image_files:
            filename = img_path.replace('\\', '/').split('/')[-1]
            student_id = filename.split('_')[0]

            feat = process_single_image(img_path)
            if feat is not None and feat[1] is not None:
                # Unique ID so DB2 does not overwrite DB1: "DB1_B_101", "DB2_B_101", ...
                unique_id = f"{db_name}_{student_id}"
                database[unique_id] = feat

    print(f"-> Done! Stored templates of {len(database)} people in the system.")
    return database


def identify_fingerprint(query_feat, database, threshold=15):
    """
    Step 2 (1:N): Identification.
    Compare one scanned fingerprint against the whole database.
    The highest-scoring entry that also passes the threshold is returned.
    Score = number of geometric matches (RANSAC inliers).
    """
    if query_feat is None or query_feat[1] is None:
        return "Image error", 0

    best_match_id = "Unknown"
    highest_score = 0

    for student_id, db_feat in database.items():
        score = match_two_fingerprints(query_feat, db_feat)
        if score > highest_score:
            highest_score = score
            best_match_id = student_id

    if highest_score < threshold:
        return "Imposter", highest_score

    return best_match_id, highest_score


def process_single_image(image_path):
    """
    Read image, preprocess (Normalization + Gaussian Blur + CLAHE) and extract SIFT.
    Returns a (keypoints, descriptors) tuple so the matcher can do geometric verification.
    """
    # Read with numpy to support non-ASCII paths
    img_array = np.fromfile(image_path, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: cannot read image at {image_path}")
        return None

    # Preprocessing: contrast stretch + denoise + local contrast equalization (CLAHE)
    normalized_img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    blurred_img = cv2.GaussianBlur(normalized_img, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_img = clahe.apply(blurred_img)

    # SIFT feature extraction
    keypoints, descriptors = _SIFT.detectAndCompute(enhanced_img, None)
    return (keypoints, descriptors)


def num_keypoints(feat):
    """Helper: number of keypoints (used for image-quality / FTA check)."""
    if feat is None or feat[1] is None:
        return 0
    return len(feat[1])


def calculate_metrics(genuine_scores, imposter_scores, thresholds):
    """Compute FAR, FRR arrays and find the EER (Equal Error Rate)."""
    far_list, frr_list = [], []
    for thresh in thresholds:
        far = sum(1 for s in imposter_scores if s >= thresh) / len(imposter_scores) if imposter_scores else 0
        frr = sum(1 for s in genuine_scores if s < thresh) / len(genuine_scores) if genuine_scores else 0
        far_list.append(far)
        frr_list.append(frr)

    # EER = the threshold where FAR and FRR are closest to each other
    differences = [abs(far - frr) for far, frr in zip(far_list, frr_list)]
    min_idx = int(np.argmin(differences))
    eer = (far_list[min_idx] + frr_list[min_idx]) / 2
    return eer, thresholds[min_idx]


def analyze_worst_matches_sift(folder_path):
    """
    Debug tool: scan and plot the 20 genuine pairs with the lowest SIFT match scores.
    Helps validate accuracy and justify the chosen threshold.
    """
    print("\n[Outlier Analysis] Cross-matching the whole dataset. Please wait...")
    results = []

    for f_id in range(101, 111):
        path_base = os.path.join(folder_path, f"{f_id}_1.tif")
        feat_base = process_single_image(path_base)

        for impression in range(2, 9):
            path_target = os.path.join(folder_path, f"{f_id}_{impression}.tif")
            feat_target = process_single_image(path_target)
            score = match_two_fingerprints(feat_base, feat_target)
            pair_name = f"{f_id}_1 vs {f_id}_{impression}"
            results.append((score, pair_name))

    results.sort(key=lambda x: x[0])
    worst_20 = results[:20]
    scores = [item[0] for item in worst_20]
    labels = [item[1] for item in worst_20]

    print("\n=== TOP 20 HARDEST GENUINE PAIRS (SIFT + RANSAC) ===")
    for i, (score, label) in enumerate(worst_20):
        print(f"{i+1:02d}. {label:.<25} {score:.0f} matches")

    print("\nBuilding bar chart...")
    plt.figure(figsize=(14, 7))
    threshold_line = 13
    colors = ['red' if s < threshold_line else 'dodgerblue' for s in scores]
    bars = plt.bar(labels, scores, color=colors, edgecolor='black')
    plt.axhline(y=threshold_line, color='red', linestyle='--', linewidth=2,
                label=f'Proposed threshold = {threshold_line} matches')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.5, f'{yval:.0f}', ha='center', va='bottom', fontsize=9)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Geometric matches (RANSAC inliers)', fontsize=12, fontweight='bold')
    plt.title('Outlier Analysis: Top 20 hardest genuine fingerprint pairs', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    folder_path = DEFAULT_FOLDER
    my_databases = [
        os.path.join(BASE_DIR, "fingerprints", "DB1_B"),
        os.path.join(BASE_DIR, "fingerprints", "DB2_B"),
        os.path.join(BASE_DIR, "fingerprints", "DB3_B"),
    ]

    while True:
        print("\n" + "=" * 50)
        print("=== FINGERPRINT VERIFICATION SYSTEM - SIFT + RANSAC ===")
        print("=" * 50)
        print("1. Run 1:1 performance evaluation (FAR, FRR, EER)")
        print("2. Start 1:N class attendance system")
        print("3. Show Outlier Analysis chart (threshold check)")
        print("4. Exit")

        choice = input("\nSelect a feature (1/2/3/4): ")

        if choice == '1':
            genuine_scores, imposter_scores, execution_times = [], [], []
            print("\n[Running] Scanning data and computing. Please wait...\n")

            # Genuine: _1 vs _2.._8 of the same finger
            for finger_id in range(101, 111):
                feat_base = process_single_image(os.path.join(folder_path, f"{finger_id}_1.tif"))
                for impression in range(2, 9):
                    start_time = time.time()
                    feat_target = process_single_image(os.path.join(folder_path, f"{finger_id}_{impression}.tif"))
                    score = match_two_fingerprints(feat_base, feat_target)
                    execution_times.append(time.time() - start_time)
                    genuine_scores.append(score)

            # Imposter: _1 of one finger vs _1 of every other finger
            for i in range(101, 111):
                feat_A = process_single_image(os.path.join(folder_path, f"{i}_1.tif"))
                for j in range(i + 1, 111):
                    feat_B = process_single_image(os.path.join(folder_path, f"{j}_1.tif"))
                    score = match_two_fingerprints(feat_A, feat_B)
                    imposter_scores.append(score)

            avg_latency = np.mean(execution_times)
            thresholds = range(0, 100)
            eer, optimal_threshold = calculate_metrics(genuine_scores, imposter_scores, thresholds)

            print("=== PERFORMANCE REPORT (1:1 VERIFICATION) ===")
            print(f"Genuine trials: {len(genuine_scores)}")
            print(f"Imposter trials: {len(imposter_scores)}")
            print(f"Genuine score (avg): {np.mean(genuine_scores):.1f}  |  Imposter score (avg): {np.mean(imposter_scores):.1f}")
            print(f"1. Average latency: {avg_latency:.4f} s/sample")
            print(f"2. Optimal threshold: {optimal_threshold} matches")
            print(f"3. Equal Error Rate (EER): {eer * 100:.2f}%")

            input("\nPress Enter to return to the main menu...")

        elif choice == '2':
            db = build_mega_database(my_databases)
            current_thresh = 13

            while True:
                print(f"\n--- SCANNER RUNNING (safety threshold: {current_thresh} matches) ---")
                print("Tip: open the folder and DRAG-AND-DROP an image file here.")
                test_f = input("Image path (or type 'exit'): ")
                if test_f.lower() == 'exit':
                    break

                t_path = test_f.strip().strip('"').strip("'")
                if not os.path.exists(t_path):
                    print("Error: invalid path or file does not exist!")
                    continue

                d_query = process_single_image(t_path)
                if num_keypoints(d_query) < 15:
                    print("\n" + "-" * 40)
                    print("ERROR (FTA): scanned image too blurry. Please rescan!")
                    print("-" * 40)
                    continue

                p_id, score = identify_fingerprint(d_query, db, threshold=current_thresh)
                print("\n" + "-" * 40)
                if p_id == "Imposter":
                    print(f"WARNING: Imposter! (best match score: {score:.0f})")
                else:
                    print(f"VERIFIED: Student {p_id} marked present! (match score: {score:.0f})")
                print("-" * 40)

        elif choice == '3':
            analyze_worst_matches_sift(folder_path)
            input("\nPress Enter to return to the menu...")
        elif choice == '4':
            print("\nThank you for using the system. Goodbye!")
            break
        else:
            print("Invalid choice, please enter 1, 2, 3 or 4.")
