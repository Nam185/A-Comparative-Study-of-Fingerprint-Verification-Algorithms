import cv2
import numpy as np
import os
import time
from skimage import feature
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root (parent of apps/)
DEFAULT_FOLDER = os.path.join(BASE_DIR, "fingerprints", "DB1_B")

# LBP configuration
LBP_RADIUS = 2
LBP_POINTS = 8 * LBP_RADIUS
LBP_GRID = 8           # Split image into 8x8 = 64 cells (block-based / spatial LBP)
IMG_SIZE = (256, 256)  # Resize to a common size so the grid cells stay aligned


def match_two_fingerprints(hist_A, hist_B):
    """
    Match two block-based LBP descriptors using the Chi-Square distance.

    IMPORTANT: hist_A/hist_B are CONCATENATED histograms of many cells (spatial LBP).
    A single GLOBAL LBP histogram cannot distinguish identity, because every
    fingerprint has very similar ridge texture (this is why the global version
    failed at ~43% EER). Splitting into cells + Chi-Square compares texture region
    by region.

    Returns a 0..100 score (higher = more similar) to stay consistent with the
    other algorithms' evaluation framework.
    """
    if hist_A is None or hist_B is None:
        return 0.0

    # Chi-Square distance: 0 = identical, larger = more different
    chi_square = 0.5 * np.sum((hist_A - hist_B) ** 2 / (hist_A + hist_B + 1e-10))

    # Convert "distance" into a 0..100 "similarity" with a decaying exponential
    similarity = float(np.exp(-chi_square)) * 100
    return round(similarity, 2)


def build_database(folder_path):
    """
    Step 1 (1:N): Enrollment.
    Returns a dictionary {person_id: LBP_grid_histogram}.
    """
    database = {}
    print("Loading fingerprints into the system (Database)...")

    for student_id in range(101, 111):
        img_path = os.path.join(folder_path, f"{student_id}_1.tif")
        feat = process_single_image(img_path)
        if feat is not None:
            database[str(student_id)] = feat

    print(f"-> Done! Stored templates of {len(database)} students.")
    return database


def identify_fingerprint(query_desc, database, threshold=15):
    """
    Step 2 (1:N): Identification.
    """
    if query_desc is None:
        return "Image error", 0

    best_match_id = "Unknown"
    highest_score = 0

    for student_id, db_desc in database.items():
        score = match_two_fingerprints(query_desc, db_desc)
        if score > highest_score:
            highest_score = score
            best_match_id = student_id

    if highest_score < threshold:
        return "Imposter", highest_score

    return best_match_id, highest_score


def process_single_image(image_path):
    """
    Preprocess (Combo 1) and extract BLOCK-BASED (spatial) LBP features.

    Pipeline: Resize -> Normalize -> Blur -> CLAHE -> uniform LBP -> split into 8x8
    cells, compute one normalized histogram per cell, then CONCATENATE them into one
    long feature vector.
    """
    img_array = np.fromfile(image_path, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    # Resize to a common size so the grid cells line up between images
    img = cv2.resize(img, IMG_SIZE)

    # Combo 1: Normalize + Gaussian Blur + CLAHE
    normalized_img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    blurred_img = cv2.GaussianBlur(normalized_img, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_img = clahe.apply(blurred_img)

    # Uniform LBP: less sensitive to rotation and noise
    lbp = feature.local_binary_pattern(enhanced_img, LBP_POINTS, LBP_RADIUS, method="uniform")
    n_bins = LBP_POINTS + 2  # fixed number of bins for the 'uniform' method

    h, w = lbp.shape
    features = []
    for gy in range(LBP_GRID):
        for gx in range(LBP_GRID):
            cell = lbp[gy * h // LBP_GRID:(gy + 1) * h // LBP_GRID,
                       gx * w // LBP_GRID:(gx + 1) * w // LBP_GRID]
            hist, _ = np.histogram(cell.ravel(), bins=n_bins, range=(0, n_bins))
            hist = hist.astype("float32")
            hist /= (hist.sum() + 1e-7)
            features.append(hist)

    return np.concatenate(features)


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


def analyze_worst_matches_lbp(folder_path):
    """Debug tool: 20 genuine pairs with the lowest LBP match scores."""
    print("\n[Outlier Analysis] Cross-matching the whole dataset. Please wait...")
    results = []

    for f_id in range(101, 111):
        path_base = os.path.join(folder_path, f"{f_id}_1.tif")
        desc_base = process_single_image(path_base)
        for impression in range(2, 9):
            path_target = os.path.join(folder_path, f"{f_id}_{impression}.tif")
            desc_target = process_single_image(path_target)
            score = match_two_fingerprints(desc_base, desc_target)
            pair_name = f"{f_id}_1 vs {f_id}_{impression}"
            results.append((score, pair_name))

    results.sort(key=lambda x: x[0])
    worst_20 = results[:20]
    scores = [item[0] for item in worst_20]
    labels = [item[1] for item in worst_20]

    print("\n=== TOP 20 HARDEST GENUINE PAIRS (block LBP) ===")
    for i, (score, label) in enumerate(worst_20):
        print(f"{i+1:02d}. {label:.<25} {score}")

    print("\nBuilding bar chart...")
    plt.figure(figsize=(14, 7))
    threshold_line = np.mean(scores) if scores else 1.0
    colors = ['red' if s < threshold_line else 'dodgerblue' for s in scores]
    bars = plt.bar(labels, scores, color=colors, edgecolor='black')
    plt.axhline(y=threshold_line, color='red', linestyle='--', linewidth=2,
                label=f'Reference threshold ({threshold_line:.1f})')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.1, f'{yval}', ha='center', va='bottom', fontsize=9)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('LBP similarity (0..100)', fontsize=12, fontweight='bold')
    plt.title('Outlier Analysis: Top 20 hardest genuine fingerprint pairs', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    folder_path = DEFAULT_FOLDER

    while True:
        print("\n" + "=" * 50)
        print("=== FINGERPRINT VERIFICATION SYSTEM - LBP (grid + Chi-Square) ===")
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
                desc_base = process_single_image(os.path.join(folder_path, f"{finger_id}_1.tif"))
                for impression in range(2, 9):
                    start_time = time.time()
                    desc_target = process_single_image(os.path.join(folder_path, f"{finger_id}_{impression}.tif"))
                    score = match_two_fingerprints(desc_base, desc_target)
                    execution_times.append(time.time() - start_time)
                    genuine_scores.append(score)

            for i in range(101, 111):
                desc_A = process_single_image(os.path.join(folder_path, f"{i}_1.tif"))
                for j in range(i + 1, 111):
                    desc_B = process_single_image(os.path.join(folder_path, f"{j}_1.tif"))
                    score = match_two_fingerprints(desc_A, desc_B)
                    imposter_scores.append(score)

            avg_latency = np.mean(execution_times)
            # LBP scores fall in a narrow range, so scan thresholds over min..max
            lo = min(genuine_scores + imposter_scores)
            hi = max(genuine_scores + imposter_scores)
            thresholds = np.linspace(lo, hi, 200)
            eer, optimal_threshold = calculate_metrics(genuine_scores, imposter_scores, thresholds)

            print("=== PERFORMANCE REPORT (1:1 VERIFICATION) ===")
            print(f"Genuine trials: {len(genuine_scores)}")
            print(f"Imposter trials: {len(imposter_scores)}")
            print(f"Genuine score (avg): {np.mean(genuine_scores):.2f}  |  Imposter score (avg): {np.mean(imposter_scores):.2f}")
            print(f"1. Average latency: {avg_latency:.4f} s/sample")
            print(f"2. Optimal threshold: {optimal_threshold:.2f}")
            print(f"3. Equal Error Rate (EER): {eer * 100:.2f}%")
            print("\n[Note] LBP is a GLOBAL TEXTURE feature with no geometric alignment")
            print("step like SIFT/ORB, so its EER is much higher. This is a valid")
            print("scientific result to discuss: SIFT/ORB suit fingerprints better than LBP.")

            input("\nPress Enter to return to the main menu...")

        elif choice == '2':
            print("\n[System] Starting the scanner and loading the database...")
            class_database = build_database(folder_path)
            attendance_threshold = 50  # similarity score (0..100); LBP needs a high threshold

            while True:
                print(f"\n--- ATTENDANCE SCANNER READY (safety threshold: {attendance_threshold}) ---")
                test_file = input("Enter fingerprint file name (e.g. 105_2.tif) or 'exit': ")
                if test_file.lower() == 'exit':
                    break

                img_test_path = os.path.join(folder_path, test_file)
                if not os.path.exists(img_test_path):
                    print("Error: file not found! Please try again.")
                    continue

                print("Analyzing fingerprint...")
                desc_query = process_single_image(img_test_path)
                if desc_query is None:
                    print("\n" + "-" * 30)
                    print("REJECTED: scanned image too blurry or corrupted!")
                    print("-" * 30)
                    continue

                predicted_id, highest_score = identify_fingerprint(desc_query, class_database, threshold=attendance_threshold)
                print("\n" + "-" * 30)
                if predicted_id == "Imposter":
                    print(f"WARNING: IMPOSTER DETECTED! (best match score only {highest_score})")
                elif predicted_id == "Image error":
                    print("ERROR: invalid fingerprint data!")
                else:
                    print(f"ATTENDANCE OK: Student ID {predicted_id} (confidence: {highest_score})")
                print("-" * 30)

        elif choice == '3':
            analyze_worst_matches_lbp(folder_path)
            input("\nPress Enter to return to the menu...")
        elif choice == '4':
            print("\nThank you for using the system. Goodbye!")
            break
        else:
            print("Invalid choice, please enter 1, 2, 3 or 4.")
