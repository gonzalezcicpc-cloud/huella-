import cv2
import numpy as np
import os

def mostrar_comparacion_clara(imgA_path, imgB_path):
    # Cargar imágenes en escala de grises
    imgA = cv2.imread(imgA_path, cv2.IMREAD_GRAYSCALE)
    imgB = cv2.imread(imgB_path, cv2.IMREAD_GRAYSCALE)

    if imgA is None or imgB is None:
        print("⚠️ No se pudieron cargar las imágenes")
        return

    # Redimensionar ambas a 300x300
    imgA_resized = cv2.resize(imgA, (300, 300))
    imgB_resized = cv2.resize(imgB, (300, 300))

    # Combinar lado a lado
    combinado = np.hstack((imgA_resized, imgB_resized))

    # Calcular diferencia absoluta y similitud
    diff = cv2.absdiff(imgA_resized, imgB_resized)
    similarity = 1 - (np.sum(diff) / (300 * 300 * 255))

    # Mostrar puntaje en consola
    print(f"Similitud visual directa: {similarity:.4f}")

    # Escribir puntaje sobre la imagen combinada
    cv2.putText(combinado, f"Similitud: {similarity:.4f}", (10, 290),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255), 2, cv2.LINE_AA)

    # Mostrar en una sola ventana clara
    cv2.imshow("Comparación final (300x300 cada una)", combinado)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Rutas de las imágenes
    imgA_path = "data/prueba.jpg"
    imgB_path = "data/prueba1.jpg"

    # Mostrar comparación clara
    mostrar_comparacion_clara(imgA_path, imgB_path)






