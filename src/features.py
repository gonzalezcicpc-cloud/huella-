import cv2
import numpy as np

def extract_features_orb(img):
    orb = cv2.ORB_create(nfeatures=3000, scaleFactor=1.2, nlevels=8, edgeThreshold=31, patchSize=31)
    kps, des = orb.detectAndCompute(img, None)
    if des is None:
        des = np.zeros((0,32), dtype=np.uint8)
    return kps, des
