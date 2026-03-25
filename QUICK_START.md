# Quick Start - Compression d'Images

## Installation

```bash
pip install pillow numpy matplotlib scikit-learn tqdm
```

## Préparation des données

```bash
# Créer dossier d'entrée (s'il n'existe pas)
mkdir -p _in

# Copier vos images dans _in/
cp /chemin/vers/images/*.jpg _in/
cp /chemin/vers/images/*.png _in/
```

## Pipeline de base

### Commande simple
```bash
python3 main.py
```

**Résultat:**
- Images redimensionnées à 224×224 pixels dans `_out/`
- Rapport JSON: `_out/rapport_compression.json`
- Barre de progression en direct

**Exemple sortie:**
```
======================================================================
Compression intelligente d'images
======================================================================
Dossier source:  _in
Dossier sortie:  _out
Taille cible:    224x224 pixels
Images trouvées: 5
Taille initiale:    1.85MB
======================================================================

████████████████████████████████████████████████████ 100% (005/005)

======================================================================
Résumé
======================================================================
Images traitées:     5/5
Taille initiale:       1.85MB
Taille comprimée:    167.47KB
Réduction totale:    91.2%
Qualité JPEG moyenne: 15.0
Similarité moyenne:   0.9839
======================================================================

Rapport sauvegardé: _out/rapport_compression.json
```

## Pipeline avec modèle ML

### Entraîner et prédire
```bash
python3 pipeline_optimise.py
```

**Phases:**
1. Compression initiale (entraînement du modèle)
2. Entraînement LinearRegression sur 6 features
3. Réapplication avec prédictions du modèle

**Résultats:**
- `_out/rapport_optimise.json` - Rapport final
- `_out/modele_predictions.png` - Graphiques ML (4 panels)
- `modele.pkl` - Modèle sauvegardé

## Analyse et visualisation

### Générer graphiques
```bash
python3 analyse.py
```

**Génère 4 graphiques:**
1. Qualité JPEG optimale par image
2. Similarité après compression
3. Réduction de taille par image
4. Résumé statistique

Sauvegarde: `_out/analyse_compression.png`

## Workflow complet (exemple)

```bash
# 1. Naviguer au dossier
cd /chemin/vers/projet_dechets

# 2. Préparer les images
mkdir -p _in
cp ~/mes_images/*.jpg _in/

# 3. Compression simple
python3 main.py

# 4. Vérifier résultats
ls -lh _out/
cat _out/rapport_compression.json

# 5. Compression avec ML
python3 pipeline_optimise.py

# 6. Analyser
python3 analyse.py

# 7. Résultat final
echo "Images 224x224 dans _out/"
echo "Rapport JSON prêt pour ML pipeline"
```

## Tester avec images exemple

```bash
# Créer 3 images de test
python3 -c "
from PIL import Image
from pathlib import Path

Path('_in').mkdir(exist_ok=True)

# Test 1: Grande image
Image.new('RGB', (800, 600), color=(200, 100, 50)).save('_in/test_large.png')

# Test 2: Petite image
Image.new('RGB', (300, 250), color=(50, 100, 200)).save('_in/test_small.png')

# Test 3: Rapport 16:9
Image.new('RGB', (1280, 720), color=(100, 200, 100)).save('_in/test_wide.jpg')

print('3 images de test créées dans _in/')
"

# Tester compression
python3 main.py
```

## Résultats attendus

### Sur 5 images déchets:
- **Taille initiale:** 1.9 MB
- **Taille finale:** ~150 KB
- **Compression:** 92%
- **Similarité:** 0.98+ (98% fidèle)
- **Format:** Toutes 224×224 pixels JPEG

### Rapport JSON structure:
```json
{
  "resume": {
    "images_traitees": 5,
    "taille_initiale_bytes": 1938420,
    "taille_finale_bytes": 150000,
    "reduction_percent": 92.3,
    "taille_cible": [224, 224]
  },
  "images": [
    {
      "fichier": "image1.png",
      "taille_orig": [800, 600],
      "taille_orig_bytes": 400000,
      "taille_redim": [224, 224],
      "taille_finale_bytes": 15000,
      "qualite_jpg": 45,
      "reduction_percent": 96.3,
      "similarite": 0.987
    }
  ]
}
```

## Configuration personnalisée

Éditer `main.py` ligne 10-12:

```python
TAILLE_CIBLE = (224, 224)              # Changer pour autre taille
QUALITES_A_TESTER = [95, 85, ..., 15]  # Ajouter/retirer qualités
SIMILARITE_MIN = 0.85                  # Augmenter pour meilleure fidélité
```

Puis relancer: `python3 main.py`

## Dépannage

**Erreur: Module not found**
```bash
pip install pillow numpy matplotlib scikit-learn tqdm
```

**Aucune image trouvée**
```bash
mkdir -p _in
ls _in/  # Vérifier qu'il y a des images
```

**Rapport pas généré**
```bash
python3 main.py  # Vérifier qu'il n'y a pas d'erreur
ls _out/rapport_compression.json  # Vérifier fichier existe
```

**Performance lente**
- Images très grosses: réduire d'abord
- Qualités à tester: réduire QUALITES_A_TESTER

## Notes

- Aspect ratio toujours préservé (padding blanc si besoin)
- Conversion systématique en JPEG
- Seuil similarité défaut: 0.85
- Rapports JSON compatible ML pipelines
- Modèle ML sauvegardé dans `modele.pkl`

---

**Temps typique:**
- 5 images: ~5-10 secondes
- 50 images: ~1 minute
- 500 images: ~10 minutes
