import cv2
import numpy as np

# ---------- Preprocesamiento ----------
def preprocess(img_gray, size=(400, 400)):
    # Normalizar tamaño
    img = cv2.resize(img_gray, size)

    # Ecualización de histograma para mejorar contraste de crestas
    img = cv2.equalizeHist(img)

    # Suavizado ligero para reducir ruido sin perder crestas
    img = cv2.GaussianBlur(img, (3, 3), 0)

    # Binarización adaptativa para resaltar crestas
    bin_img = cv2.adaptiveThreshold(img, 255,
                                    cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY, 15, 5)

    return img, bin_img

# ---------- Extracción de características (aprox. minucias con ORB) ----------
def extract_descriptors(img_gray):
    if img_gray is None:
        return None, None, None

    # Preprocesar
    pre_img, bin_img = preprocess(img_gray)

    # ORB para keypoints y descriptores robustos en patrones de crestas
    orb = cv2.ORB_create(
        nfeatures=2000,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=31,
        firstLevel=0,
        WTA_K=2,
        scoreType=cv2.ORB_HARRIS_SCORE,
        patchSize=31,
        fastThreshold=20
    )
    keypoints, descriptors = orb.detectAndCompute(pre_img, None)

    return keypoints, descriptors, pre_img

# ---------- Comparación de huellas ----------
def compare_descriptors(descA, descB):
    if descA is None or descB is None or len(descA) == 0 or len(descB) == 0:
        return 0.0, []

    # Matcher de fuerza bruta con Hamming para ORB
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(descA, descB, k=2)

    # Filtro de Lowe (ratio test) para eliminar falsos positivos
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    # Puntaje de similitud como proporción de matches buenos
    similarity = len(good) / max(len(descA), len(descB))
    return similarity, good

# ---------- Comparación completa (paths) ----------
def compare_fingerprints(imgA_path, imgB_path):
    imgA = cv2.imread(imgA_path, cv2.IMREAD_GRAYSCALE)
    imgB = cv2.imread(imgB_path, cv2.IMREAD_GRAYSCALE)

    if imgA is None or imgB is None:
        return None, None, 0.0, "Error al cargar imágenes"

    kpA, descA, preA = extract_descriptors(imgA)
    kpB, descB, preB = extract_descriptors(imgB)

    similarity, good_matches = compare_descriptors(descA, descB)

    # Generar visual de comparación clara (lado a lado) + texto de similitud
    preA_resized = cv2.resize(preA, (400, 400))
    preB_resized = cv2.resize(preB, (400, 400))
    combined = np.hstack((preA_resized, preB_resized))

    cv2.putText(combined, f"Similitud (IA): {similarity:.4f}", (10, 390),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255), 2, cv2.LINE_AA)

    # También puedes generar visual de coincidencias (opcional, no usado en UI)
    # match_vis = cv2.drawMatches(preA, kpA, preB, kpB, good_matches, None,
    #                             flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    return combined, good_matches, similarity, "OK"
