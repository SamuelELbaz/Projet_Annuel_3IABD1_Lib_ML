"""
Compression et redimensionnement intelligent d'images pour ML
PA 3IABD 2025-26 - Classification de déchets
"""

import os
import json
import numpy as np
from PIL import Image
from pathlib import Path

TAILLE_CIBLE = (224, 224)
QUALITES_A_TESTER = [95, 85, 75, 65, 55, 45, 35, 25, 15]
SIMILARITE_MIN = 0.95


def get_size_format(b, factor=1024, suffix="B"):
    for unit in ["", "K", "M", "G", "T", "P", "E", "Z"]:
        if b < factor:
            return f"{b:6.2f}{unit}{suffix}"
        b /= factor
    return f"{b:6.2f}Y{suffix}"


def dir_size(dir_path):
    return sum(p.stat().st_size for p in dir_path.iterdir() if p.is_file())


def print_loading_bar(val_act, nb_total, bar_length=100):
    val_act = min(val_act, nb_total - 1)
    progress = (val_act + 1) / nb_total if nb_total else 0
    filled = int(bar_length * progress)
    empty = bar_length - filled
    bar = ("█" * filled) + ("░" * empty)
    percent = round(progress * 100)
    if val_act >= nb_total:
        percent = 100
        bar = "█" * bar_length
    print(f"{bar} {percent:03d}% ({(val_act+1):03d}/{nb_total:03d})", end="\r", flush=True)


def redimensionner_avec_padding(img, taille_cible=TAILLE_CIBLE):
    largeur_cible, hauteur_cible = taille_cible
    img_width, img_height = img.size
    
    ratio = min(largeur_cible / img_width, hauteur_cible / img_height)
    nouvelle_width = int(img_width * ratio)
    nouvelle_height = int(img_height * ratio)
    
    img_redim = img.resize((nouvelle_width, nouvelle_height), Image.LANCZOS)
    img_final = Image.new('RGB', taille_cible, (255, 255, 255))
    
    offset_x = (largeur_cible - nouvelle_width) // 2
    offset_y = (hauteur_cible - nouvelle_height) // 2
    img_final.paste(img_redim, (offset_x, offset_y))
    
    return img_final


def calculer_similarite(img1, img2):
    if img1.size != img2.size:
        return 0.0
    arr1 = np.array(img1).astype(float)
    arr2 = np.array(img2).astype(float)
    mse = np.mean((arr1 - arr2) ** 2)
    if mse == 0:
        return 1.0
    psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    if psnr >= 50:
        return 1.0
    if psnr <= 20:
        return 0.0
    return (psnr - 20) / 30.0


def trouver_qualite_optimale(img, chemin_sortie, qualites=QUALITES_A_TESTER, similarite_min=SIMILARITE_MIN):
    meilleure_qualite = qualites[0]
    meilleure_taille = float('inf')
    meilleure_similarite = 1.0
    
    img_ref = img.convert("RGB") if img.mode != 'RGB' else img
    
    for qualite in qualites:
        try:
            img_ref.save(chemin_sortie, quality=qualite, optimize=True)
        except OSError:
            img_ref.convert("RGB").save(chemin_sortie, quality=qualite, optimize=True)
        
        img_comprimee = Image.open(chemin_sortie)
        similarite = calculer_similarite(img_ref, img_comprimee)
        img_comprimee.close()
        
        taille_bytes = chemin_sortie.stat().st_size
        
        if similarite >= similarite_min and taille_bytes < meilleure_taille:
            meilleure_qualite = qualite
            meilleure_taille = taille_bytes
            meilleure_similarite = similarite
    
    try:
        img_ref.save(chemin_sortie, quality=meilleure_qualite, optimize=True)
    except OSError:
        img_ref.convert("RGB").save(chemin_sortie, quality=meilleure_qualite, optimize=True)
    
    return meilleure_qualite, meilleure_taille, meilleure_similarite


def compresser_image(image_name, redimensionner=True, taille_cible=TAILLE_CIBLE, quality=90, output_dir="_out"):
    image_path = Path(image_name)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img = Image.open(image_path)
    
    taille_orig = img.size
    taille_orig_bytes = image_path.stat().st_size
    
    if redimensionner:
        img = redimensionner_avec_padding(img, taille_cible)
    
    stem = image_path.stem
    out_name = f"{stem}_processed.jpg"
    saved_path = out_dir / out_name

    quality, taille_finale_bytes, similarite = trouver_qualite_optimale(
        img, saved_path, QUALITES_A_TESTER, SIMILARITE_MIN
    )

    ratio_compression = taille_finale_bytes / taille_orig_bytes if taille_orig_bytes > 0 else 1.0
    reduction = (1 - ratio_compression) * 100

    infos = {
        'fichier': image_path.name,
        'taille_orig': taille_orig,
        'taille_orig_bytes': taille_orig_bytes,
        'taille_redim': img.size,
        'taille_finale_bytes': taille_finale_bytes,
        'qualite_jpg': quality,
        'ratio_compression': ratio_compression,
        'reduction_percent': reduction,
        'similarite': similarite,
        'chemin_sortie': str(saved_path)
    }

    return infos


def compresser_dossier(in_dir, output_dir="_out", taille_cible=TAILLE_CIBLE, sauvegarder_rapport=True):
    valid_ext = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    files = [f for f in in_dir.iterdir() if f.is_file() and f.suffix.lower() in valid_ext]
    nb_total = len(files)
    
    if nb_total == 0:
        print(f"Aucune image trouvée dans {in_dir}")
        return {}

    init_dir_size = dir_size(in_dir)
    
    print(f"\n{'='*70}")
    print(f"Compression intelligente d'images")
    print(f"{'='*70}")
    print(f"Dossier source:  {in_dir}")
    print(f"Dossier sortie:  {output_dir}")
    print(f"Taille cible:    {taille_cible[0]}x{taille_cible[1]} pixels")
    print(f"Images trouvées: {nb_total}")
    print(f"Taille initiale:  {get_size_format(init_dir_size)}")
    print(f"{'='*70}\n")

    print_loading_bar(0, nb_total)
    resultats = []

    for i, img_path in enumerate(files):
        try:
            infos = compresser_image(
                img_path,
                redimensionner=True,
                taille_cible=taille_cible,
                output_dir=output_dir
            )
            resultats.append(infos)
        except Exception as e:
            print(f"Erreur avec {img_path.name}: {e}")
        
        print_loading_bar(i + 1, nb_total)

    print()
    
    if resultats:
        new_dir_size = dir_size(Path(output_dir))
        reduction_totale = (1 - new_dir_size / init_dir_size) * 100 if init_dir_size > 0 else 0
        
        print(f"\n{'='*70}")
        print(f"Résumé")
        print(f"{'='*70}")
        print(f"Images traitées:     {len(resultats)}/{nb_total}")
        print(f"Taille initiale:     {get_size_format(init_dir_size)}")
        print(f"Taille comprimée:    {get_size_format(new_dir_size)}")
        print(f"Réduction totale:    {reduction_totale:.1f}%")
        
        qualites = [r['qualite_jpg'] for r in resultats]
        similarites = [r['similarite'] for r in resultats]
        
        print(f"Qualité JPEG moyenne: {sum(qualites)/len(qualites):.1f}")
        print(f"Similarité moyenne:   {sum(similarites)/len(similarites):.4f}")
        print(f"{'='*70}\n")
        
        if sauvegarder_rapport:
            rapport_path = Path(output_dir) / "rapport_compression.json"
            with open(rapport_path, 'w', encoding='utf-8') as f:
                rapport = {
                    'resume': {
                        'images_traitees': len(resultats),
                        'taille_initiale_bytes': init_dir_size,
                        'taille_finale_bytes': new_dir_size,
                        'reduction_percent': reduction_totale,
                        'taille_cible': taille_cible
                    },
                    'images': resultats
                }
                json.dump(rapport, f, indent=2, ensure_ascii=False)
            print(f"Rapport sauvegardé: {rapport_path}\n")
        
        return {
            'nb_images': len(resultats),
            'taille_init': init_dir_size,
            'taille_finale': new_dir_size,
            'reduction_percent': reduction_totale
        }
    
    return {}


if __name__ == "__main__":
    dossier_source = Path("_in")
    
    if not dossier_source.exists():
        print(f"Créez le dossier '{dossier_source}' et y mettez vos images")
        exit(1)
    
    compresser_dossier(
        dossier_source,
        output_dir="_out",
        taille_cible=TAILLE_CIBLE,
        sauvegarder_rapport=True
    )
