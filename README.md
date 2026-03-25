# Compression d'Images pour ML - PA 3IABD

Preprocessing pour normaliser dataset déchets:
- Redimensionne à **224×224 pixels** exactement
- Compresse intelligemment avec qualité JPEG optimale
- Génère rapports JSON détaillés

## Fichiers

- `main.py` - Pipeline de compression
- `modele_lineaire.py` - Modèle ML pour qualité JPEG
- `pipeline_optimise.py` - Pipeline avec modèle
- `analyse.py` - Visualisations

## Utilisation

### Compression simple
```bash
python3 main.py
```

### Avec modèle ML
```bash
python3 pipeline_optimise.py
```

### Visualiser résultats
```bash
python3 analyse.py
```

## Dossiers

```
_in/       Ajouter vos images ici
_out/      Résultats générés:
  ├── *.jpg                   (224×224 redimensionnées)
  ├── rapport_compression.json
  └── modele_predictions.png
```

## Configuration

Dans `main.py`:
```python
TAILLE_CIBLE = (224, 224)
QUALITES_A_TESTER = [95, 85, 75, 65, 55, 45, 35, 25, 15]
SIMILARITE_MIN = 0.85
```

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

**Projet:** PA 3IABD 2025-26 - Classification de déchets

---

*Date: 25 mars 2026*
