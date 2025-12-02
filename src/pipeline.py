import cv2
import numpy as np
from .preprocess import enhance_fingerprint
from .features import extract_features_orb
from .quality import assess_quality
from .liveness import liveness_score
from .siamese import siamese_compare

def process_pair(img_path_a: str, img_path_b: str, use_deep=False):
    imgA = cv2.imread(img_path_a, cv2.IMREAD_GRAYSCALE)
    imgB = cv2.imread(img_path_b, cv2.IMREAD_GRAYSCALE)

    qa = assess_quality(imgA)
    qb = assess_quality(imgB)

    liveA = liveness_score(imgA)
    liveB = liveness_score(imgB)

    enhA = enhance_fingerprint(imgA)
    enhB = enhance_fingerprint(imgB)

    if use_deep:
        score = siamese_compare(enhA, enhB)
        method = "siamese"
        matches_vis = None
    else:
        kpsA, desA = extract_features_orb(enhA)
        kpsB, desB = extract_features_orb(enhB)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(desA, desB)
        matches = sorted(matches, key=lambda m: m.distance)
        score = 1.0 / (np.mean([m.distance for m in matches[:50]]) + 1e-6)
        method = "orb"
        matches_vis = cv2.drawMatches(enhA, kpsA, enhB, kpsB, matches[:30], None, flags=2)

    return {
        "quality": {"A": qa, "B": qb},
        "liveness": {"A": liveA, "B": liveB},
        "similarity_score": float(score),
        "method": method,
        "matches_vis": matches_vis,
    }
