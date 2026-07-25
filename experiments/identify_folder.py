"""1:N identification on a folder of probe images — with imposter rejection.

Enrolls a database (default DB1_B, template = impression _1 of fingers 101..110),
then runs every image in a probe folder against it and reports, per image:
    best matching ID, its score, and ACCEPT (identified) or REJECT (imposter).

This adds the missing imposter-filtering step: drop a genuine database fingerprint
(should be ACCEPTED) and an outside image downloaded from the internet / a different
finger (should be REJECTED) into the probe folder and compare.

Usage:
    python experiments/identify_folder.py                         # SIFT, DB1_B, fingerprints/probe/
    python experiments/identify_folder.py --algo sift --db DB1_B --probe fingerprints/probe
    python experiments/identify_folder.py --algo orb --threshold 12
"""
import os
import sys
import glob
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.io_utils import BASE_DIR, read_gray, img_path
from core.preprocessing import c1
from core.features import sift_features, orb_features, minutiae_features
from core.matching import match
from core import minutiae_native as MN

# algo -> (extractor, match_fn, default 1:N accept threshold)
ALGOS = {
    "sift": (sift_features, lambda a, b: match(a, b, scoring="S3"), 13),
    "orb": (orb_features, lambda a, b: match(a, b, scoring="S3"), 10),
    "minutiae": (minutiae_features, lambda a, b: match(a, b, scoring="S3"), 10),
    "minutiae-native": (MN.minutiae_features, MN.match, 8),
}
IMG_EXT = (".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff")


def enroll(db, extractor, fingers=range(101, 111)):
    """Build the gallery: {finger_id: features of its _1 impression}."""
    templates = {}
    for fid in fingers:
        img = read_gray(img_path(db, fid, 1))
        if img is not None:
            templates[fid] = extractor(c1(img))
    return templates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=list(ALGOS), default="sift")
    ap.add_argument("--db", default="DB1_B")
    ap.add_argument("--probe", default=os.path.join(BASE_DIR, "fingerprints", "probe"))
    ap.add_argument("--threshold", type=float, default=None)
    args = ap.parse_args()

    extractor, match_fn, default_thr = ALGOS[args.algo]
    thr = args.threshold if args.threshold is not None else default_thr

    os.makedirs(args.probe, exist_ok=True)
    probes = [p for p in sorted(glob.glob(os.path.join(args.probe, "*")))
              if p.lower().endswith(IMG_EXT)]
    if not probes:
        print(f"No probe images found in: {args.probe}")
        print("Drop images there to test, e.g.:")
        print(f"  - a genuine database fingerprint (copy of {args.db}/105_2.tif)  -> should be ACCEPTED")
        print("  - an outside fingerprint downloaded from the internet            -> should be REJECTED")
        return

    print(f"\n=== 1:N identification | algo={args.algo} | gallery={args.db} (10 fingers) "
          f"| accept threshold={thr} ===\n")
    templates = enroll(args.db, extractor)

    print(f"{'probe image':40s} {'best ID':>8s} {'score':>7s}   verdict")
    print("-" * 75)
    n_accept = n_reject = 0
    for p in probes:
        img = read_gray(p)
        if img is None:
            print(f"{os.path.basename(p):40s} {'--':>8s} {'--':>7s}   UNREADABLE")
            continue
        feat = extractor(c1(img))
        best_id, best_score = None, -1.0
        for fid, tfeat in templates.items():
            s = match_fn(feat, tfeat)
            if s > best_score:
                best_score, best_id = s, fid
        accepted = best_score >= thr
        verdict = f"ACCEPT -> ID {best_id}" if accepted else "REJECT (imposter)"
        n_accept += accepted
        n_reject += not accepted
        print(f"{os.path.basename(p):40s} {best_id:>8d} {best_score:>7.0f}   {verdict}")

    print("-" * 75)
    print(f"Summary: {len(probes)} probes -> {n_accept} accepted, {n_reject} rejected as imposter.")
    print("(A genuine enrolled finger should ACCEPT; an unknown/outside finger should REJECT.)")


if __name__ == "__main__":
    main()
