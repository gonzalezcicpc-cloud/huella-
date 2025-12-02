import cv2
import numpy as np
from skimage.restoration import denoise_bilateral

def enhance_fingerprint(img: np.ndarray) -> np.ndarray:
    img = (img / 255.0).astype(np.float32)
    den = denoise_bilateral(img, sigma_color=0.05, sigma_spatial=3, channel_axis=None)

    thetas = np.linspace(0, np.pi, 8, endpoint=False)
    accum = np.zeros_like(den)
    for th in thetas:
        ksize = 15
        kern = cv2.getGaborKernel((ksize, ksize), 4.0, th, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        resp = cv2.filter2D(den, cv2.CV_32F, kern)
        accum = np.maximum(accum, resp)

    accum = cv2.normalize(accum, None, 0, 1, cv2.NORM_MINMAX)
    accum_u8 = (accum * 255).astype(np.uint8)
    enh = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(accum_u8)
    return enh
