"""
Pipeline optimise avec modele lineaire
"""

import os
import json
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
from modele_lineaire import ModeleQualiteJPEG

TAILLE_CIBLE = (224, 224)
QUALITES_A_TESTER = [95, 85, 75, 65, 55, 45, 35, 25, 15]
SIMILARITE_MIN = 0.85


def redimensionner_avec_padding(img, taille_cible):
    img.thumbnail(taille_cible, Image.Resampling.LANCZOS)
    padding_img = Image.new('RGB', taille_cible, (255, 255, 255))
    offset = ((taille_cible[0] - img.size[0]) // 2, (taille_cible[1] - img.size[1]) // 2)
    padding_img.paste(img, offset)
    return padding_img


def calculer_similarite(img1, img2):
    arr1 = np.array(img1, dtype=np.float32)
    arr2 = np.array(img2, dtype=np.float32)
    mse = np.mean((arr1 - arr2) ** 2) / (255 ** 2)
    return max(0, min(1, 1 - mse))


def compresser_image_avec_modele(path, modele, output_dir):
    
    img_original = Image.open(path).convert('RGB')
    taille_orig = img_original.size
    taille_orig_bytes = os.path.getsize(path)
    
    img_redim = redimensionner_avec_padding(img_original, TAILLE_CIBLE)
    
    qualite_predite = modele.predire_qualite(path)
    if qualite_predite is None:
        qualite_predite = 45
    
    nom_base = Path(path).stem
    path_sortie = os.path.join(output_dir, f"{nom_base}_processed.jpg")
    img_redim.save(path_sortie, 'JPEG', quality=qualite_predite, optimize=True)
    
    img_comprimee = Image.open(path_sortie).convert('RGB')
    similarite = calculer_similarite(img_redim, img_comprimee)
    
    taille_finale = os.path.getsize(path_sortie)
    reduction = (1 - taille_finale / taille_orig_bytes) * 100
    
    return {
        'fichier': Path(path).name,
        'taille_orig': list(taille_orig),
        'taille_orig_bytes': int(taille_orig_bytes),
        'taille_redim': list(TAILLE_CIBLE),
        'taille_finale_bytes': int(taille_finale),
        'qualite_jpg': int(qualite_predite),
        'reduction_percent': float(reduction),
        'similarite': float(similarite),
        'chemin_sortie': path_sortie
    }


def main():
    Path('_out').mkdir(exist_ok=True)
    
    print("\n" + "="*70)
    print("Compression optimisee avec modele lineaire")
    print("="*70)
    
    print("\nPhase 1: Compression initiale")
    print("-" * 70)
    
    rapport_path = '_out/rapport_compression.json'
    if not Path(rapport_path).exists():
        os.system('/usr/bin/python3 main.py')
    
    print("\nPhase 2: Entrainement du modele")
    print("-" * 70)
    
    modele = ModeleQualiteJPEG()
    
    if not modele.entrainer_sur_rapport(rapport_path):
        print("Erreur lors de l'entrainement")
        return
    
    modele.sauvegarder()
    
    print("\nPhase 3: Compression optimisee")
    print("-" * 70)
    
    for f in Path('_out').glob('*.jpg'):
        f.unlink()
    
    images = [f for f in Path('_in').iterdir() if f.suffix.lower() in ['.jpg', '.png', '.jpeg']]
    
    resultats = []
    taille_init_total = 0
    
    for img_path in tqdm(images, desc="Traitement"):
        try:
            result = compresser_image_avec_modele(str(img_path), modele, '_out')
            resultats.append(result)
            taille_init_total += result['taille_orig_bytes']
        except Exception as e:
            print(f"Erreur {img_path.name}: {e}")
    
    taille_finale_total = sum(r['taille_finale_bytes'] for r in resultats)
    reduction_totale = (1 - taille_finale_total / taille_init_total * 100) * 100 if taille_init_total > 0 else 0
    qualite_moyenne = float(np.mean([r['qualite_jpg'] for r in resultats]))
    similarite_moyenne = float(np.mean([r['similarite'] for r in resultats]))
    
    rapport_final = {
        'resume': {
            'images_traitees': len(resultats),
            'taille_initiale_bytes': int(taille_init_total),
            'taille_finale_bytes': int(taille_finale_total),
            'reduction_percent': float(reduction_totale),
            'taille_cible': list(TAILLE_CIBLE),
            'modele_utilise': 'LinearRegression',
            'qualite_moyenne': float(qualite_moyenne),
            'similarite_moyenne': float(similarite_moyenne)
        },
        'images': resultats
    }
    
    with open('_out/rapport_optimise.json', 'w') as f:
        json.dump(rapport_final, f, indent=2)
    
    print("\n" + "="*70)
    print("Resume final")
    print("="*70)
    print(f"Images traitees:       {len(resultats)}")
    print(f"Taille initiale:       {taille_init_total/1024:.2f} KB")
    print(f"Taille comprimee:      {taille_finale_total/1024:.2f} KB")
    print(f"Reduction totale:      {reduction_totale:.1f}%")
    print(f"Qualite JPEG moyenne:  {qualite_moyenne:.1f}")
    print(f"Similarite moyenne:    {similarite_moyenne:.4f}")
    print("="*70)
    print(f"Rapport: _out/rapport_optimise.json")
    print(f"Graphs:  _out/modele_predictions.png")
    print("="*70 + "\n")
    
    print("Generation des graphiques...")
    modele.visualiser_predictions()


if __name__ == "__main__":
    main()
