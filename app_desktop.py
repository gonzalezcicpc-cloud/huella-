import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QFileDialog, QLineEdit, QMessageBox
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

from ia_lofoscopia import compare_fingerprints

# -----------------------------
# Interfaz principal IDTROCONIS
# -----------------------------
class IDTROCONIS_UI(QWidget):
    # ... (todo tu código PyQt5 aquí, igual como lo tienes)
    pass

# -----------------------------
# Ventana de Login
# -----------------------------
class LoginWindow(QWidget):
    # ... (todo tu código PyQt5 aquí, igual como lo tienes)
    pass

# -----------------------------
# Arranque de la aplicación
# -----------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    login = LoginWindow()
    login.show()
    sys.exit(app.exec_())
