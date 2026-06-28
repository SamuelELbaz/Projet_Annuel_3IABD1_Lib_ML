"""
Script de pretraitement des images pour classification de dechets.
"""

import random
from pathlib import Path
from PIL import Image

# Configuration 

SOURCE_DIR     = Path("images")   # Dossier source avec les 3 sous-dossiers
OUTPUT_DIR     = Path("dataset")  # Dossier de sortie

# Les 4 configurations de pretraitement
CONFIGS = [
    {"nom": "32x32_gris", "taille": (32, 32),  "mode": "L"},
    {"nom": "32x32_rgb",  "taille": (32, 32),  "mode": "RGB"},
    {"nom": "64x64_gris", "taille": (64, 64),  "mode": "L"},
    {"nom": "64x64_rgb",  "taille": (64, 64),  "mode": "RGB"},
]

# Proportions du decoupage
RATIO_TRAIN = 0.70
RATIO_VAL   = 0.15
RATIO_TEST  = 0.15   

# Seed pour que le decoupage soit reproductible d'une execution a l'autre.

SEED = 42

EXTENSIONS_VALIDES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

# Decoupage train / val / test

def decouper(fichiers: list, seed: int) -> dict:
    """
    Melange et divise une liste de fichiers en train / val / test.
    Le seed garantit que le decoupage est identique a chaque execution.

    Retourne un dict {"train": [...], "val": [...], "test": [...]}.
    """
    fichiers = list(fichiers)   # copie pour ne pas modifier l'original
    random.seed(seed)
    random.shuffle(fichiers)

    n       = len(fichiers)
    n_train = int(n * RATIO_TRAIN)
    n_val   = int(n * RATIO_VAL)
    # le reste va dans test (evite les arrondis qui font perdre des images)

    return {
        "train": fichiers[:n_train],
        "val":   fichiers[n_train : n_train + n_val],
        "test":  fichiers[n_train + n_val :],
    }

# Traitement d'une image

def traiter_image(source: Path, destination: Path,
                  taille: tuple, mode: str) -> bool:
    """
    Ouvre source, convertit en mode (L ou RGB),
    redimensionne en taille, sauvegarde en destination.
    Retourne True si succes, False si erreur.
    """
    try:
        with Image.open(source) as img:
            img_conv  = img.convert(mode)
            img_redim = img_conv.resize(taille, Image.LANCZOS)
            img_redim.save(destination)
        return True
    except Exception as e:
        print(f"    [ERREUR] Erreur sur {source.name} : {e}")
        return False

# Traitement principal

def pretraiter(source: Path, sortie: Path) -> None:

    # Recuperer les classes (sous-dossiers du dossier source)
    classes = sorted([
        d.name for d in source.iterdir()
        if d.is_dir()
    ])

    if not classes:
        print("[ERREUR] Aucun sous-dossier trouve dans le dossier source.")
        return

    print(f"Classes detectees : {classes}")
    print(f"Configs           : {[c['nom'] for c in CONFIGS]}")
    print(f"Decoupage         : {RATIO_TRAIN:.0%} train / "
          f"{RATIO_VAL:.0%} val / {RATIO_TEST:.0%} test  (seed={SEED})")
    print()

    # Etape 1 : decouper les fichiers (une seule fois, avant les configs)

    decoupages = {}   # { classe: {"train": [...], "val": [...], "test": [...]} }

    print("-" * 50)
    print("Decoupage des fichiers")
    print("-" * 50)

    for classe in classes:
        dossier_classe = source / classe
        fichiers = sorted([
            f for f in dossier_classe.iterdir()
            if f.is_file() and f.suffix.lower() in EXTENSIONS_VALIDES
        ])

        if not fichiers:
            print(f"  [ATTENTION]  Aucun fichier valide dans '{classe}', classe ignoree.")
            continue

        split = decouper(fichiers, seed=SEED)
        decoupages[classe] = split

        print(f"  {classe:20s} : "
              f"{len(fichiers)} images  ->  "
              f"train={len(split['train'])}  "
              f"val={len(split['val'])}  "
              f"test={len(split['test'])}")

    print()

    # Etape 2 : generer les 4 configs

    total_ok     = 0
    total_erreur = 0

    for config in CONFIGS:
        nom    = config["nom"]
        taille = config["taille"]
        mode   = config["mode"]

        print("-" * 50)
        print(f"Config : {nom}  ({taille[0]}x{taille[1]}, mode={mode})")
        print("-" * 50)

        for classe, split in decoupages.items():
            for ensemble, fichiers in split.items():

                # Creer le dossier de destination si besoin
                dossier_dest = sortie / nom / ensemble / classe
                dossier_dest.mkdir(parents=True, exist_ok=True)

                ok = erreur = 0
                for fichier in fichiers:
                    dest = dossier_dest / fichier.name
                    if traiter_image(fichier, dest, taille, mode):
                        ok += 1
                    else:
                        erreur += 1

                total_ok     += ok
                total_erreur += erreur

            print(f"  {classe:20s} : "
                  f"train={len(split['train'])}  "
                  f"val={len(split['val'])}  "
                  f"test={len(split['test'])}")

        print()

    # Resume final

    print("=" * 50)
    print(f"[OK]  Images traitees avec succes : {total_ok}")
    print(f"[ERR]  Erreurs                     : {total_erreur}")
    print(f"[>] Dataset sauvegarde dans     : {sortie.resolve()}")
    print()
    print("Structure produite :")
    for config in CONFIGS:
        print(f"  {sortie / config['nom']}/")
        for ensemble in ("train", "val", "test"):
            n = sum(
                len(list((sortie / config["nom"] / ensemble / c).glob("*")))
                for c in decoupages
            )
            print(f"    {ensemble}/  ({n} images)")


# Point d'entree

if __name__ == "__main__":
    if not SOURCE_DIR.exists():
        print(f"[ERREUR] Dossier source introuvable : {SOURCE_DIR.resolve()}")
        print("   -> Modifie la variable SOURCE_DIR en haut du script.")
    else:
        pretraiter(SOURCE_DIR, OUTPUT_DIR)