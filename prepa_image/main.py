import csv
import numpy as np
from PIL import Image
from pathlib import Path

TAILLE_CIBLE = (224, 224)
QUALITES_A_TESTER = [95, 85, 75, 65, 55, 45, 35, 25, 15]
SIMILARITE_MIN = 0.95


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
    meilleure_similarite = 0.0
    
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
    
    if meilleure_taille == float('inf'):
        meilleure_qualite = qualites[0]
        img_ref.save(chemin_sortie, quality=meilleure_qualite, optimize=True)
        meilleure_taille = chemin_sortie.stat().st_size
        img_test = Image.open(chemin_sortie)
        meilleure_similarite = calculer_similarite(img_ref, img_test)
        img_test.close()
    else:
        img_ref.save(chemin_sortie, quality=meilleure_qualite, optimize=True)
    
    return meilleure_qualite, meilleure_taille, meilleure_similarite


def extraire_features(image_path):
    img = Image.open(image_path)
    arr = np.array(img)
    
    r_mean = np.mean(arr[:,:,0])
    g_mean = np.mean(arr[:,:,1])
    b_mean = np.mean(arr[:,:,2])
    
    return r_mean, g_mean, b_mean


def generer_dataset_csv(images_dir, output_csv):
    images_dir = Path(images_dir)
    
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["r_mean", "g_mean", "b_mean", "label"])
        
        for img_path in images_dir.iterdir():
            if img_path.suffix.lower() not in [".jpg", ".png", ".jpeg"]:
                continue
            
            r, g, b = extraire_features(img_path)
            
            if "dechet" in img_path.name.lower() or "organique" in img_path.name.lower():
                label = 1
            else:
                label = 0
            
            writer.writerow([r, g, b, label])


def compresser_image(image_name, output_dir="_out"):
    image_path = Path(image_name)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img = Image.open(image_path)
    img = redimensionner_avec_padding(img, TAILLE_CIBLE)
    
    stem = image_path.stem
    out_name = f"{stem}_processed.jpg"
    saved_path = out_dir / out_name

    trouver_qualite_optimale(img, saved_path, QUALITES_A_TESTER, SIMILARITE_MIN)


def compresser_dossier(in_dir="_in", output_dir="_out"):
    import shutil
    
    out_path = Path(output_dir)
    if out_path.exists():
        shutil.rmtree(out_path)
    out_path.mkdir(parents=True, exist_ok=True)
    
    valid_ext = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    files = [f for f in Path(in_dir).iterdir() if f.is_file() and f.suffix.lower() in valid_ext]
    
    if not files:
        print(f"Aucuneimage dans {in_dir}")
        return

    for img_path in files:
        try:
            compresser_image(img_path, output_dir)
        except Exception as e:
            print(f"Erreur {img_path.name}: {e}")


if __name__ == "__main__":
    dossier_source = Path("_in")
    
    if not dossier_source.exists():
        print(f"Créez le dossier '_in' et y mettez vos images")
        exit(1)
    
    compresser_dossier()
    generer_dataset_csv("_out", "dataset.csv")
    print("Compression terminée")
