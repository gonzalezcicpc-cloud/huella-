import numpy as np
import cv2

def liveness_score(img):
    high = cv2.Laplacian(img, cv2.CV_32F)
    hf_energy = float(np.mean(np.abs(high)))

    glcm = cv2.equalizeHist(img)
    edges = cv2.Canny(glcm, 50, 150)
    edge_density = float(edges.mean())

    score = 0.6 * (hf_energy / (hf_energy + 1)) + 0.4 * (edge_density / (edge_density + 1))
    return score
