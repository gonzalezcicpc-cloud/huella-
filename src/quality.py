import numpy as np
import cv2

def assess_quality(img):
    contrast = float(img.std())
    sharpness = float(cv2.Laplacian(img, cv2.CV_64F).var())
    blur = cv2.GaussianBlur(img, (5,5), 0)
    noise = float(np.mean(np.abs(img.astype(np.float32) - blur.astype(np.float32))))
    return {"contrast": contrast, "sharpness": sharpness, "noise": noise}
