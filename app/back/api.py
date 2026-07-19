"""
api.py — Serveur Flask pour la classification de dechets.
"""

import sys
import io
import tempfile
import os
from pathlib import Path
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np

BASE_DIR        = Path(__file__).parent
MODELE_PMC      = BASE_DIR / "pmc" / "meilleur_modele.txt"
MODELE_LINEAIRE = BASE_DIR / "model_lineaire" / "notebook" / "meilleur_modele_lineaire.txt"
MODELE_SVM      = BASE_DIR / "svm" / "svm_final_dechets.txt"
RBF_DIR         = BASE_DIR / "rbf"
PMC_LIB         = BASE_DIR / "pmc"
LINMODEL_LIB    = BASE_DIR / "model_lineaire" / "notebook"
SVM_LIB         = BASE_DIR / "svm"

sys.path.insert(0, str(PMC_LIB))
from pmc import PMC

sys.path.insert(0, str(LINMODEL_LIB))
from linmodel_lib import LinearModel

sys.path.insert(0, str(SVM_LIB))
from svm_lib import SVM_OVR

os.chdir(str(RBF_DIR))
sys.path.insert(0, str(RBF_DIR))
from inferance_rbf import predire as rbf_predire
os.chdir(str(BASE_DIR))

app = Flask(__name__)

CLASSES     = ["compost", "dechets_chimiques", "recyclable"]
TARGET_SIZE = (32, 32)

print("Chargement des modeles...")

model_pmc      = PMC.load(str(MODELE_PMC))
model_lineaire = LinearModel.load(str(MODELE_LINEAIRE))
model_svm      = SVM_OVR.load(str(MODELE_SVM))

print("Modeles prets.")


def pretraiter_image(image_bytes):
    img    = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img    = img.resize(TARGET_SIZE, Image.LANCZOS)
    pixels = np.array(img, dtype=np.float64).flatten() / 255.0
    return pixels


def scores_vers_dict(scores):
    return {CLASSES[i]: round(float(scores[i]), 4) for i in range(len(CLASSES))}


@app.route("/")
def index():
    return open(BASE_DIR / "index.html", encoding="utf-8").read()


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"erreur": "Aucune image recue"}), 400

    fichier = request.files["image"]
    if fichier.filename == "":
        return jsonify({"erreur": "Fichier vide"}), 400

    try:
        image_bytes = fichier.read()
        x           = pretraiter_image(image_bytes)

        scores_pmc = model_pmc.forward(x)
        scores_lin = model_lineaire.forward(x)
        scores_svm = model_svm.decision_function(x.reshape(1, -1))[0]

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name
        try:
            rbf_result = rbf_predire(tmp_path)
            scores_rbf = [rbf_result["scores"][c] for c in CLASSES]
            classe_rbf = rbf_result["classe"]
        finally:
            os.remove(tmp_path)

        resultats = {
            "PMC": {
                "classe": CLASSES[int(np.argmax(scores_pmc))],
                "scores": scores_vers_dict(scores_pmc),
            },
            "Lineaire": {
                "classe": CLASSES[int(np.argmax(scores_lin))],
                "scores": scores_vers_dict(scores_lin),
            },
            "SVM": {
                "classe": CLASSES[int(np.argmax(scores_svm))],
                "scores": scores_vers_dict(scores_svm),
            },
            "RBF": {
                "classe": classe_rbf,
                "scores": {c: round(float(s), 4)
                           for c, s in rbf_result["scores"].items()},
            },
        }

        return jsonify({"resultats": resultats})

    except Exception as e:
        return jsonify({"erreur": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)