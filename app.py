import os
import cv2
from flask import Flask, render_template, request
from ia_lofoscopia import compare_fingerprints

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'results'

# Crear carpetas si no existen
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# Ruta principal: portal de bienvenida
@app.route("/")
def home():
    return render_template("index.html")

# Ruta para subir y comparar huellas
@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        fileA = request.files.get("imgA")
        fileB = request.files.get("imgB")

        if not fileA or not fileB or fileA.filename == "" or fileB.filename == "":
            return render_template("upload.html", error="⚠️ Debes subir ambas huellas")

        pathA = os.path.join(app.config['UPLOAD_FOLDER'], fileA.filename)
        pathB = os.path.join(app.config['UPLOAD_FOLDER'], fileB.filename)
        fileA.save(pathA)
        fileB.save(pathB)

        combined, matches, similarity, status = compare_fingerprints(pathA, pathB)
        if status != "OK" or combined is None:
            return render_template("upload.html", error="Error al procesar las huellas")

        result_path = os.path.join(app.config['RESULT_FOLDER'], "resultado.jpg")
        cv2.imwrite(result_path, combined)

        return render_template("upload.html",
                               similarity=f"{similarity:.4f}",
                               result_image=result_path)

    return render_template("upload.html")

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8080)




