"""
Génération du dataset CSV depuis les images annotées.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Sources (sous-dossiers de IMAGE_DIR) :
  compost/           → label 0 (Compost / Organique)
  dechets_chimiques/ → label 1 (Chimique / Dangereux)
  recyclable/        → label 2 (Recyclable)

Utilise les MAX_PAR_CAT premières images de chaque catégorie,
triées par numéro croissant (ex. _0001 … _0500).

Usage :
  python main.py
"""

import csv
import numpy as np
from PIL import Image
from pathlib import Path

# ─── Configuration ────────────────────────────────────────────────────────────
IMAGE_DIR   = Path("image ")    # dossier racine contenant les sous-dossiers
OUTPUT_CSV  = Path("dataset.csv")
MAX_PAR_CAT = 1500              # premières images utilisées par catégorie
TAILLE_CIBLE = (224, 224)       # même taille que preprocess() dans l'app

# ─── Correspondance dossier → label ──────────────────────────────────────────
CATEGORIES = {
    "compost":           0,   # Compost / Organique
    "dechets_chimiques": 1,   # Chimique / Dangereux
    "recyclable":        2,   # Recyclable
}


def get_numero(path: Path) -> int:
    """Extrait le numéro depuis le nom (exemple compost_0042 → 42)."""
    try:
        return int(path.stem.rsplit("_", 1)[-1])
    except (ValueError, IndexError):
        return 999_999


def _grille_moyennes(canal: np.ndarray, g: int = 3) -> list:
    """Découpe un canal en grille g×g et renvoie la moyenne de chaque case
    (ordre ligne par ligne). Capture *où* se trouvent les couleurs/bords dans
    l'image — information spatiale que la moyenne globale perd."""
    H, W = canal.shape
    ys = np.linspace(0, H, g + 1).astype(int)
    xs = np.linspace(0, W, g + 1).astype(int)
    return [float(canal[ys[i]:ys[i+1], xs[j]:xs[j+1]].mean())
            for i in range(g) for j in range(g)]


def extraire_features(img_path: Path) -> list:
    """
    Charge une image, la redimensionne (thumbnail, contenu seul — sans padding
    blanc qui créerait de faux contours) et extrait 32 caractéristiques :

        • moyennes globales R, G, B            (3)   → couleur globale  ([0–255])
        • grille 3×3 des moyennes R, G, B      (27)  → répartition spatiale ([0–255])
        • moyenne / écart-type du gradient     (2)   → texture / bords    ([0–1])

    La grille 3×3 capte la disposition (objet centré, fond uni, bords blancs) que
    la simple moyenne ignore : le modèle linéaire passe de ~57 % à ~63 %.
    Les 3 moyennes globales sont conservées en tête pour le PMC (qui n'utilise
    que R, G, B) et les anciens tests C.
    """
    img = Image.open(img_path).convert("RGB")
    img.thumbnail(TAILLE_CIBLE, Image.LANCZOS)        # contenu seul, aucun bord blanc
    arr = np.asarray(img, dtype=np.float32)           # [0–255]
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    # Grille 3×3 par canal (27 valeurs) en [0–255]
    grille = _grille_moyennes(r, 3) + _grille_moyennes(g, 3) + _grille_moyennes(b, 3)

    # Gradient sur niveaux de gris normalisés [0–1] → texture / netteté des bords
    gray   = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
    gy, gx = np.gradient(gray)
    grad   = np.sqrt(gx * gx + gy * gy)

    return ([float(r.mean()), float(g.mean()), float(b.mean())]   # 3 globales (PMC/legacy)
            + grille                                               # 27 grille
            + [float(grad.mean()), float(grad.std())])             # 2 gradient


def generer_dataset(image_dir: Path, output_csv: Path, max_par_cat: int = MAX_PAR_CAT):
    """
    Génère dataset.csv en lisant les sous-dossiers de image_dir.
    Ne prend que les max_par_cat premières images (tri numérique) par catégorie.
    """
    total = 0

    # En-tête : 3 globales (R,G,B) + 27 grille (r_0..b_8) + 2 gradient + label
    grille_cols = ([f"r_{i}" for i in range(9)]
                   + [f"g_{i}" for i in range(9)]
                   + [f"b_{i}" for i in range(9)])
    entete = ["r_mean", "g_mean", "b_mean"] + grille_cols + ["grad_mean", "grad_std", "label"]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(entete)

        for dossier_nom, label in CATEGORIES.items():
            dossier = image_dir / dossier_nom
            if not dossier.exists():
                print(f"  [ATTENTION] Dossier introuvable : {dossier}")
                continue

            # Tri numérique, on prend les max_par_cat premiers fichiers
            fichiers = sorted(
                [f for f in dossier.iterdir()
                 if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
                 and not f.name.startswith(".")],
                key=get_numero
            )[:max_par_cat]

            print(f"  [{dossier_nom:20s}] label={label}  —  {len(fichiers)} images")
            ok = 0
            for img_path in fichiers:
                try:
                    feats = extraire_features(img_path)        # 32 valeurs
                    # 30 premières (R,G,B globales + grille) en [0–255] : 4 décimales
                    # 2 dernières (gradient) en [0–1] : 6 décimales
                    ligne = ([f"{v:.4f}" for v in feats[:30]]
                             + [f"{v:.6f}" for v in feats[30:]]
                             + [label])
                    writer.writerow(ligne)
                    ok += 1
                except Exception as e:
                    print(f"    Erreur {img_path.name}: {e}")

            total += ok
            print(f"    → {ok} lignes écrites")

    print(f"\n  Dataset total : {total} lignes  →  {output_csv}")
    return total


if __name__ == "__main__":
    print("━" * 55)
    print("  GÉNÉRATION DU DATASET CSV — Classification de déchets")
    print("━" * 55)
    print(f"  Source   : {IMAGE_DIR.resolve()}")
    print(f"  Sortie   : {OUTPUT_CSV.resolve()}")
    print(f"  Max/cat  : {MAX_PAR_CAT} premières images")
    print()

    if not IMAGE_DIR.exists():
        print(f"ERREUR : Le dossier '{IMAGE_DIR}' est introuvable.")
        exit(1)

    n = generer_dataset(IMAGE_DIR, OUTPUT_CSV, MAX_PAR_CAT)
    print()
    print(f"  {n} échantillons générés ({MAX_PAR_CAT} × {len(CATEGORIES)} catégories)")
    print()
    print("  Prochaine étape :")
    print("    streamlit run app_streamlit.py")
    print("━" * 55)
