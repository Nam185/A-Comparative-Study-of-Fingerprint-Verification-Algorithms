"""Evaluation protocols: 1:1 EER and 1:N Rank-1 / identification rate.

Fingers are IDs 101..110, impressions 1..8.
Genuine 1:1 pairs  = _1 vs _2.._8 of the same finger (70 / DB).
Imposter 1:1 pairs = _1 vs _1 of different fingers     (45 / DB).
"""
import numpy as np

FINGERS = range(101, 111)
IMPRESSIONS = range(2, 9)


def collect_scores(feat_fn, match_fn):
    """feat_fn(finger_id, impression) -> feature (caller caches). Returns (genuine, imposter)."""
    genuine, imposter = [], []
    for fid in FINGERS:
        base = feat_fn(fid, 1)
        for imp in IMPRESSIONS:
            genuine.append(match_fn(base, feat_fn(fid, imp)))
    for i in FINGERS:
        a = feat_fn(i, 1)
        for j in FINGERS:
            if j > i:
                imposter.append(match_fn(a, feat_fn(j, 1)))
    return genuine, imposter


def eer(genuine, imposter):
    """Equal Error Rate (%) and the threshold where FAR == FRR (closest)."""
    g = np.asarray(genuine, float)
    im = np.asarray(imposter, float)
    cands = np.unique(np.concatenate([g, im]))
    best = (2.0, 0.0, 0.0)  # (|far-frr|, eer, threshold)
    for t in cands:
        far = np.mean(im >= t) if len(im) else 0.0
        frr = np.mean(g < t) if len(g) else 0.0
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2 * 100, float(t))
    return best[1], best[2]


def rank1_and_idrate(feat_fn, match_fn, threshold):
    """1:N: enroll _1 of every finger; query with _2.._8 (genuine queries).

    Rank-1 accuracy   = fraction of queries whose top match is the correct finger.
    Identification rate = fraction correct AND top score >= threshold.
    """
    templates = {fid: feat_fn(fid, 1) for fid in FINGERS}
    total = correct_rank1 = correct_at_thr = 0
    for true_fid in FINGERS:
        for imp in IMPRESSIONS:
            q = feat_fn(true_fid, imp)
            best_id, best_score = None, -1.0
            for tid, tfeat in templates.items():
                s = match_fn(q, tfeat)
                if s > best_score:
                    best_score, best_id = s, tid
            total += 1
            if best_id == true_fid:
                correct_rank1 += 1
                if best_score >= threshold:
                    correct_at_thr += 1
    return (correct_rank1 / total * 100, correct_at_thr / total * 100, total)
