import cv2
import numpy as np

# Cargar ambas imágenes
imgA = cv2.imread("data/prueba.jpg", cv2.IMREAD_GRAYSCALE)
imgB = cv2.imread("data/prueba1.jpg", cv2.IMREAD_GRAYSCALE)

# Verificar carga
if imgA is None or imgB is None:
    print("⚠️ No se pudieron cargar las imágenes")
    exit()

# Redimensionar a mismo tamaño si difieren
if imgA.shape != imgB.shape:
    imgB = cv2.resize(imgB, (imgA.shape[1], imgA.shape[0]))

# Calcular diferencia absoluta
diff = cv2.absdiff(imgA, imgB)
similarity = 1 - (np.sum(diff) / (imgA.shape[0]*imgA.shape[1]*255))

print(f"Similitud entre imágenes: {similarity:.4f}")

# Mostrar lado a lado
combined = np.hstack((imgA, imgB))
cv2.imshow("Imagen A (izq) vs Imagen B (der)", combined)

# Mostrar diferencia
cv2.imshow("Diferencia absoluta", diff)

cv2.waitKey(0)
cv2.destroyAllWindows()
