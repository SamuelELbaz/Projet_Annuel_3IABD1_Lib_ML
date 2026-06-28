"""
Charge une configuration du dataset prepare par pretraitement.py
et produit des tableaux numpy prets a passer au PMC.
"""

import numpy as np
from pathlib import Path
from PIL import Image

DATASET_DIR = Path("dataset")


def charger_ensemble(dossier: Path, classes: list) -> tuple:
    """
    Charge toutes les images d'un ensemble (train, val ou test)

    Retourne (X, y) :
        X : (n, n_features) float64, pixels normalises dans [0, 1]
        y : (n,)            int32,   indice de la classe
    """
    X_list = []
    y_list = []

    for idx_classe, classe in enumerate(classes):
        dossier_classe = dossier / classe
        if not dossier_classe.exists():
            print(f" Dossier introuvable : {dossier_classe}")
            continue

        fichiers = sorted([
            f for f in dossier_classe.iterdir()
            if f.is_file() and f.suffix.lower() in
            {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
        ])

        for fichier in fichiers:
            try:
                with Image.open(fichier) as img:
                    # Aplatir en vecteur 1D et normaliser dans [0, 1]
                    pixels = np.array(img, dtype=np.float64).flatten() / 255.0
                    X_list.append(pixels)
                    y_list.append(idx_classe)
            except Exception as e:
                print(f" Erreur sur {fichier.name} : {e}")

    if not X_list:
        return np.empty((0, 0)), np.empty(0, dtype=np.int32)

    return (
        np.array(X_list, dtype=np.float64),
        np.array(y_list, dtype=np.int32),
    )


def charger_config(nom_config: str, verbose: bool = True) -> tuple:
    """
    Charge les trois ensembles (train, val, test) pour une configuration.
    """
    dossier_config = DATASET_DIR / nom_config

    if not dossier_config.exists():
        raise FileNotFoundError(
            f"Config '{nom_config}' introuvable dans {DATASET_DIR.resolve()}.\n"
            f"Lancez d'abord pretraitement.py."
        )

    # Detecter les classes depuis le dossier train
    dossier_train = dossier_config / "train"
    classes = sorted([
        d.name for d in dossier_train.iterdir()
        if d.is_dir()
    ])

    if not classes:
        raise ValueError(f"Aucune classe trouvee dans {dossier_train}")

    if verbose:
        print(f"Config     : {nom_config}")
        print(f"Classes    : {classes}")
        print(f"Chargement en cours...")

    X_train, y_train = charger_ensemble(dossier_config / "train", classes)
    X_val,   y_val   = charger_ensemble(dossier_config / "val",   classes)
    X_test,  y_test  = charger_ensemble(dossier_config / "test",  classes)

    if verbose:
        print(f"\nDimensions :")
        print(f"  X_train : {X_train.shape}  y_train : {y_train.shape}")
        print(f"  X_val   : {X_val.shape}    y_val   : {y_val.shape}")
        print(f"  X_test  : {X_test.shape}   y_test  : {y_test.shape}")
        print(f"  n_features = {X_train.shape[1]}")
        print(f"  Pixels normalises dans [0, 1] : "
              f"min={X_train.min():.3f}  max={X_train.max():.3f}")
        print()

        # Repartition par classe
        print(f"Repartition par classe :")
        for idx, classe in enumerate(classes):
            n_tr = np.sum(y_train == idx)
            n_va = np.sum(y_val   == idx)
            n_te = np.sum(y_test  == idx)
            print(f"  {classe:20s} : train={n_tr}  val={n_va}  test={n_te}")
        print()

    return X_train, y_train, X_val, y_val, X_test, y_test, classes


def labels_onehot(y: np.ndarray, n_classes: int,
                  bipolaire: bool = True) -> np.ndarray:
    """
    Convertit un vecteur de labels entiers en matrice one-hot.

    Exemples avec 3 classes et y[i] = 1 :
        bipolaire=True  -> [-1, +1, -1]
        bipolaire=False -> [ 0,  1,  0]
    """
    valeur_negatif = -1.0 if bipolaire else 0.0
    valeur_positif =  1.0

    Y = np.full((len(y), n_classes), valeur_negatif, dtype=np.float64)
    for i, label in enumerate(y):
        Y[i, label] = valeur_positif
    return Y


if __name__ == "__main__":

    CONFIG = "32x32_rgb"   
    # 32x32_rgb, 64x64_rgb, 32x32_gris, 64x64_gris