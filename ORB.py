import cv2
import numpy as np
import os
import time
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_FOLDER = os.path.join(BASE_DIR, "fingerprints", "DB1_B")

# Create ORB once. More features helps because fingerprints have few sharp corners.
_ORB = cv2.ORB_create(nfeatures=1500)


def match_two_fingerprints(feat_A, feat_B, ratio=0.8):
    """
    Match two fingerprints with ORB + Hamming distance + Lowe's Ratio + RANSAC.

    feat_A, feat_B are (keypoints, descriptors) tuples.
    Score = number of GEOMETRICALLY CONSISTENT matches (RANSAC inliers).

    ORB uses binary descriptors, so distances must be measured with NORM_HAMMING.
    After the ratio test removes noise, RANSAC keeps only matches that follow one
    consistent geometric transform, removing the accidental matches of impostors.
    """
    if feat_A is None or feat_B is None:
        return 0.0

    kp_A, desc_A = feat_A
    kp_B, desc_B = feat_B
    if desc_A is None or desc_B is None or len(desc_A) < 2 or len(desc_B) < 2:
        return 0.0

    # ORB requires NORM_HAMMING
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(desc_A, desc_B, k=2)
    good_matches = []
    for pair in matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio * n.distance:
                good_matches.append(m)

    if len(good_matches) < 4:
        return float(len(good_matches))

    src_pts = np.float32([kp_A[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_B[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if mask is None:
        return 0.0

    inliers = int(mask.sum())
    return float(inliers)


def build_database(folder_path):
    """
    Step 1 (1:N): Enrollment.
    Use impression _1.tif of 10 fingers as the enrolled templates.
    Returns a dictionary {person_id: (keypoints, descriptors)}.
    """
    database = {}
    print("Loading fingerprints into the system (Database)...")

    for student_id in range(101, 111):
        img_path = os.path.join(folder_path, f"{student_id}_1.tif")
        feat = process_single_image(img_path)
        if feat is not None and feat[1] is not None:
            database[str(student_id)] = feat

    print(f"-> Done! Stored templates of {len(database)} students.")
    return database


def identify_fingerprint(query_feat, database, threshold=15):
    """
    Step 2 (1:N): Identification.
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
    Read image, preprocess (Normalization + Gaussian Blur + CLAHE) and extract ORB.
    Returns a (keypoints, descriptors) tuple.
    """
    img_array = np.fromfile(image_path, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: cannot read image at {image_path}")
        return None

    normalized_img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    blurred_img = cv2.GaussianBlur(normalized_img, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_img = clahe.apply(blurred_img)

    keypoints, descriptors = _ORB.detectAndCompute(enhanced_img, None)
    return (keypoints, descriptors)


def num_keypoints(feat):
    if feat is None or feat[1] is None:
        return 0
    return len(feat[1])


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


def analyze_worst_matches_orb(folder_path):
    """Debug tool: 20 genuine pairs with the lowest ORB match scores."""
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

    print("\n=== TOP 20 HARDEST GENUINE PAIRS (ORB + RANSAC) ===")
    for i, (score, label) in enumerate(worst_20):
        print(f"{i+1:02d}. {label:.<25} {score:.0f} matches")

    print("\nBuilding bar chart...")
    plt.figure(figsize=(14, 7))
    threshold_line = 10
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

    while True:
        print("\n" + "=" * 50)
        print("=== FINGERPRINT VERIFICATION SYSTEM - ORB + RANSAC ===")
        print("=" * 50)
        print("1. Run 1:1 performance evaluation (FAR, FRR, EER)")
        print("2. Start 1:N class attendance system")
        print("3. Show Outlier Analysis chart (threshold check)")
        print("4. Exit")

        choice = input("\nSelect a feature (1/2/3/4): ")

        if choice == '1':
            genuine_scores, imposter_scores, execution_times = [], [], []
            print("\n[Running] Scanning data and computing. Please wait...\n")

            for finger_id in range(101, 111):
                feat_base = process_single_image(os.path.join(folder_path, f"{finger_id}_1.tif"))
                for impression in range(2, 9):
                    start_time = time.time()
                    feat_target = process_single_image(os.path.join(folder_path, f"{finger_id}_{impression}.tif"))
                    score = match_two_fingerprints(feat_base, feat_target)
                    execution_times.append(time.time() - start_time)
                    genuine_scores.append(score)

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
            print("\n[System] Starting the scanner and loading the database...")
            class_database = build_database(folder_path)
            attendance_threshold = 10

            while True:
                print(f"\n--- ATTENDANCE SCANNER READY (safety threshold: {attendance_threshold} matches) ---")
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
                    print("REJECTED: scanned image too blurry or corrupted, no fingerprint found!")
                    print("-" * 30)
                    continue

                predicted_id, highest_score = identify_fingerprint(desc_query, class_database, threshold=attendance_threshold)
                print("\n" + "-" * 30)
                if predicted_id == "Imposter":
                    print(f"WARNING: IMPOSTER DETECTED! (best match score only {highest_score:.0f})")
                elif predicted_id == "Image error":
                    print("ERROR: invalid fingerprint data!")
                else:
                    print(f"ATTENDANCE OK: Student ID {predicted_id} (match score: {highest_score:.0f})")
                print("-" * 30)

        elif choice == '3':
            analyze_worst_matches_orb(folder_path)
            input("\nPress Enter to return to the menu...")
        elif choice == '4':
            print("\nThank you for using the system. Goodbye!")
            break
        else:
            print("Invalid choice, please enter 1, 2, 3 or 4.")
