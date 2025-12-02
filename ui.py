import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog
)
from PyQt5.QtGui import QPixmap, QImage

from ia_lofoscopia import compare_fingerprints


class IDTROCONIS_UI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IDTROCONIS - Sistema de Comparación Biométrica")
        self.setGeometry(100, 100, 1000, 700)  # Ventana grande

        # Etiqueta principal
        self.label = QLabel("Selecciona dos huellas para comparar")
        self.label.setStyleSheet("font-size: 18px;")
        self.label.setMinimumSize(900, 500)  # Área amplia para visualizar el resultado

        # Botones
        self.btn1 = QPushButton("Seleccionar Huella de Referencia")
        self.btn2 = QPushButton("Seleccionar Huella a Verificar")
        self.btnRun = QPushButton("Ejecutar Comparación")
        self.btnSave = QPushButton("Guardar Resultado")  # opcional

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.btn1)
        layout.addWidget(self.btn2)
        layout.addWidget(self.btnRun)
        layout.addWidget(self.btnSave)
        self.setLayout(layout)

        # Variables
        self.imgA = None
        self.imgB = None
        self.last_result = None  # imagen combinada del último resultado

        # Conexiones
        self.btn1.clicked.connect(self.load_imgA)
        self.btn2.clicked.connect(self.load_imgB)
        self.btnRun.clicked.connect(self.run_comparacion)
        self.btnSave.clicked.connect(self.save_result)

    def load_imgA(self):
        path, _ = QFileDialog.getOpenFileName(self, "Seleccionar Huella de Referencia", "", "Images (*.jpg *.png)")
        if path:
            self.imgA = path
            self.label.setText(f"Huella de Referencia cargada:\n{path}")

    def load_imgB(self):
        path, _ = QFileDialog.getOpenFileName(self, "Seleccionar Huella a Verificar", "", "Images (*.jpg *.png)")
        if path:
            self.imgB = path
            self.label.setText(f"Huella a Verificar cargada:\n{path}")

    def run_comparacion(self):
        if self.imgA and self.imgB:
            combined, matches, similarity, status = compare_fingerprints(self.imgA, self.imgB)

            if status != "OK" or combined is None:
                self.label.setText("⚠️ Error al procesar las huellas")
                self.setWindowTitle("IDTROCONIS - Error")
                return

            # Mostrar resultado en la interfaz
            h, w = combined.shape
            qimg = QImage(combined.data, w, h, w, QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(qimg)
            self.label.setPixmap(pixmap)
            self.label.setScaledContents(True)

            # Guardar último resultado en memoria para exportar si se desea
            self.last_result = combined.copy()

            # Actualizar título con puntaje
            self.setWindowTitle(f"IDTROCONIS - Similitud (IA): {similarity:.4f}")
        else:
            self.label.setText("⚠️ Carga ambas huellas primero")

    def save_result(self):
        if self.last_result is None:
            self.label.setText("⚠️ No hay resultado para guardar. Ejecuta una comparación primero.")
            return

        # Crear carpeta results si no existe
        os.makedirs("results", exist_ok=True)
        save_path = os.path.join("results", "comparacion_idtroconis.jpg")

        # Guardar imagen combinada
        cv2.imwrite(save_path, self.last_result)
        self.setWindowTitle(f"IDTROCONIS - Resultado guardado en: {save_path}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = IDTROCONIS_UI()
    window.show()
    sys.exit(app.exec_())




