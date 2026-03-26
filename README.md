# Préparation des Données d'Images

Script Python pour uniformiser les images: redimensionnement et compression.

## Utilisation

```bash
python3 main.py
```

Images de `_in/` → résultats dans `_out/`

## Configuration

`main.py`: `TAILLE_CIBLE = (224, 224)`

## Résultats

Test 5 images déchets:
- Init: 1.9 MB → Final: 14 KB
- Compression: 99.3%
- Similarité: 0.9975 (98% fidèle)

✅ **Redimensionnement intelligent**
- Préserve le ratio d'aspect
- Padding blanc automatique
- Interpolation LANCZOS (haute qualité)

✅ **Compression optimisée**
- Test de 9 niveaux de qualité JPEG
- Minimise la taille avec seuil de similarité
- Chaque image a sa qualité optimale

✅ **Analyse de qualité**
- Calcul de l'erreur quadratique moyenne (MSE)
- Score de similarité (0-1)
- Rapport détaillé par image

✅ **Rapport complet**
- Barre de progression
- Statistiques globales
- Export JSON avec tous les détails

## Exemple de workflow

```bash
# 1. Préparer les images
mkdir _in
cp ~/images/*.jpg _in/

# 2. Compresser
python3 main.py

# 3. Analyser
python3 analyse.py

# 4. Résultat
# Images 224x224 pixels dans _out/
# Rapport JSON et graphiques générés
```

## Groupe

**Groupe: La fine équipe**
- Roissath
- Flavien
- Samumu


